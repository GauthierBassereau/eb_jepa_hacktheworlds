import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.vis_utils import (
    analyze_distances,
    create_comparison_gif,
    plot_losses,
    save_decoded_frames,
    save_gif,
    show_images,
)

logger = get_logger(__name__)

planner_name_map = {
    "cem": "CEMPlanner",
    "mppi": "MPPIPlanner",
}
objective_name_map = {
    "repr_dist": "ReprTargetDistMPCObjective",
    "repr_dist_collision": "ReprDistCollisionMPCObjective",
    "probe_pos": "ProbePositionMPCObjective",
    "learned_value": "LearnedValueMPCObjective",
}


def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    prober=None,
    cfg=None,
):
    """
    Evaluate the model's unrolling capabilities by comparing unrolled predictions to ground truth.
    """
    env = env_creator()
    env.reset()
    device = next(model.parameters()).device
    normalizer = (
        loader.dataset.normalizer if hasattr(loader.dataset, "normalizer") else None
    )
    agent = GCAgent(
        model=model, plan_cfg=None, normalizer=normalizer, env=env, loc_prober=prober
    )
    mse_values = []
    position_mse_values = []
    unroll_times = []
    loader_iter = iter(loader)

    for idx in tqdm(
        range(num_samples), desc="Evaluating unroll", disable=cfg.logging.tqdm_silent
    ):
        try:
            x, a, loc, wall_x, door_y = next(loader_iter)
        except StopIteration:
            logger.warning(
                f"Loader exhausted after {idx} samples (requested {num_samples})"
            )
            break

        x = x.to(device)
        a = a.to(device)
        with torch.no_grad():
            obs_init = x[:, :, 0:1]  # B C T H W
            start_time = time.time()
            predicted_states = agent.unroll(obs_init, a, repeat_batch=False)[
                :, :, :-1
            ]  # discard last predicted state
            end_time = time.time()
            unroll_times.append(end_time - start_time)
            rand_predicted_states = agent.unroll(
                obs_init, torch.randn_like(a), repeat_batch=False
            )[
                :, :, :-1
            ]  # B D T H W
            # To ensure independence across timesteps when encoding the sequence, batchify it
            # There is no independence between timesteps when using GroupNorm, even in eval mode
            B, C, T, H, W = x.shape
            gt_encoded = (
                model.encode(x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2))
                .squeeze(2)
                .unflatten(dim=0, sizes=(B, -1))
                .permute(0, 2, 1, 3, 4)
            )
            latent_mse = (
                ((gt_encoded - predicted_states) ** 2).mean(dim=(1, 3, 4)).cpu().numpy()
            )  # B T
            mse_values.append(latent_mse)

            if prober:
                gt_decoded = agent.decode_loc_to_pixel(gt_encoded, wall_x, door_y)
                pred_decoded = agent.decode_loc_to_pixel(
                    predicted_states, wall_x, door_y
                )
                rand_pred_decoded = agent.decode_loc_to_pixel(
                    rand_predicted_states, wall_x, door_y
                )  # B T H W C
                gt_frames = agent.normalizer.unnormalize_state(
                    x.permute(0, 2, 1, 3, 4)
                ).permute(0, 1, 3, 4, 2)
                gt_frames = (
                    (gt_frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                )  # B T H W C uint8

                # Decode positions from predicted_states and compute MSE with ground truth
                B_probe, D_probe, T_probe, H_probe, W_probe = predicted_states.shape
                pred_positions = (
                    prober.apply_head(predicted_states).permute(0, 2, 1).cpu()
                )  # B T 2
                gt_positions = loc.permute(0, 2, 1)  # B T 2
                position_mse = (
                    ((pred_positions - gt_positions.cpu()) ** 2)
                    .mean(dim=-1)
                    .cpu()
                    .numpy()
                )  # B T
                position_mse_values.append(position_mse)

                create_comparison_gif(
                    gt_frames,
                    pred_decoded,
                    rand_pred_decoded,
                    gt_dec=gt_decoded,
                    save_path=f"{eval_folder}/b{idx}.gif",
                )
    all_mse_values = np.vstack(mse_values)  # Shape: [num_batches, T]
    mean_mse_per_timestep = np.mean(all_mse_values, axis=0)  # Shape: [T]
    std_mse_per_timestep = np.std(all_mse_values, axis=0)  # Shape: [T]
    avg_unroll_time = np.mean(unroll_times)
    results = {}
    for t in range(mean_mse_per_timestep.shape[0]):
        results[f"val_rollout/mean_mse/{t}"] = mean_mse_per_timestep[t]
        results[f"val_rollout/std_mse/{t}"] = std_mse_per_timestep[t]

    # Log position MSE if prober was used
    if len(position_mse_values) > 0:
        all_position_mse_values = np.vstack(
            position_mse_values
        )  # Shape: [num_batches, T]
        mean_position_mse_per_timestep = np.mean(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        std_position_mse_per_timestep = np.std(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        for t in range(mean_position_mse_per_timestep.shape[0]):
            results[f"val_rollout/mean_pos_mse/{t}"] = mean_position_mse_per_timestep[t]
            results[f"val_rollout/std_pos_mse/{t}"] = std_position_mse_per_timestep[t]

    results["avg_unroll_time"] = avg_unroll_time

    pd.DataFrame([results]).to_csv(f"{eval_folder}/eval.csv", index=None)
    return results


def _diagnose_world_model(agent, env, obs):
    """One-shot probe: does the world-model predict the right 1-step move for
    each cardinal action? Prints, per action, the probe-decoded predicted
    position vs what the env would actually do (move / wall-stay). Localises the
    0%-success failure: model (wrong/zero predicted deltas) vs optimisation."""
    device = agent.device
    cs = getattr(env, "cell_size", 1)  # maze-only attr; 1 for grid-free envs (two_rooms)
    obs_tensor = (
        env.normalizer.normalize_state(
            obs.detach().clone().to(dtype=torch.float32, device=device)
        )
        .unsqueeze(0)
        .unsqueeze(2)
    )  # 1 C 1 H W
    cur_cell = env.agent_cell.copy()
    true_pix = env.dot_position.detach().cpu().numpy()
    with torch.no_grad():
        enc0 = agent.model.encode(obs_tensor)
        pos0 = (
            agent.normalizer.unnormalize_location(
                agent.loc_prober.apply_head(enc0.float()).permute(0, 2, 1)
            )[0, 0]
            .cpu()
            .numpy()
        )
    logger.info(
        f"[DIAG] agent_cell={cur_cell.tolist()} goal_cell={env.goal_cell.tolist()} "
        f"| true_pix={true_pix} | probe(curr)={pos0}"
    )
    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    for name, (dr, dc) in dirs.items():
        act = torch.tensor(
            [[float(dr * cs)], [float(dc * cs)]], device=device
        ).unsqueeze(0)  # [1, 2, 1]
        with torch.no_grad():
            pred = agent.unroll(obs_tensor, act, repeat_batch=False)  # [1,D,1,H,W]
            pred_pos = (
                agent.normalizer.unnormalize_location(
                    agent.loc_prober.apply_head(pred.float()).permute(0, 2, 1)
                )[0, -1]
                .cpu()
                .numpy()
            )
        nr, nc = int(cur_cell[0] + dr), int(cur_cell[1] + dc)
        is_path = (
            0 <= nr < env.maze_height
            and 0 <= nc < env.maze_width
            and int(env.maze_grid[nr, nc].item()) == 1
        )
        env_result = f"move->cell({nr},{nc})" if is_path else "WALL(stay)"
        logger.info(
            f"[DIAG] act {name:5s} delta_pix={(pred_pos - pos0).round(2)} "
            f"pred_pix={pred_pos.round(2)} | env: {env_result}"
        )


### Main planning eval loop ###
def main_eval(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes=10,
    n_parallel=1,
    loader=None,
    prober=None,
    value_head=None,
):
    plan_cfg = OmegaConf.create(plan_cfg)

    if n_parallel > 1:
        return _main_eval_parallel(
            plan_cfg, model, env_creator, eval_folder, num_episodes, n_parallel, prober
        )

    env = env_creator()
    env.reset()

    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=env.normalizer,
        loc_prober=prober,
        env=env,
        value_head=value_head,
    )
    logger.info(f"Agent created with planner {agent.planner.__class__.__name__}")
    logger.info(f"Planning with {plan_cfg=}")

    task_spec = plan_cfg.get("task_specification", {})
    waypoint_mode = task_spec.get("waypoint_mode", False)
    waypoint_spacing = task_spec.get("waypoint_spacing", 4)
    waypoint_reach = task_spec.get("waypoint_reach_cells", 1.5) * getattr(env, "cell_size", 1)
    waypoint_action_prior = task_spec.get("waypoint_action_prior", False)
    stall_escape = task_spec.get("stall_escape", False)
    stall_patience = int(task_spec.get("stall_patience", 3))
    stop_on_success = task_spec.get("stop_on_success", False)
    if waypoint_mode:
        logger.info(
            f"Waypoint planning ON (spacing={waypoint_spacing} cells, "
            f"reach<{waypoint_reach:.1f}px, stop_on_success={stop_on_success})"
        )

    successes = []
    distances = []
    episode_times = []
    episode_observations = []
    episode_infos = []

    for ep in range(num_episodes):
        episode_start_time = time.time()
        ep_folder = eval_folder / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)
        if agent.decode_each_iteration:
            ep_plan_vis_dir = ep_folder / "plan_vis"
            os.makedirs(ep_plan_vis_dir, exist_ok=True)

        if plan_cfg.task_specification.goal_source == "dset":
            obs_slice, a, loc, _, _ = next(iter(loader))
            # obs, init_loc = obs_slice[0], loc[0]
            # goal_img, goal_loc = obs_slice[-1], loc[-1]  # [C, H, W] uint8 tensor
            # env.set_goal(goal_img) # Set goal in the environment
        elif plan_cfg.task_specification.goal_source == "random_state":
            obs, info = env.reset()  # [C, H, W] uint8 tensor
            obs, reward, done, truncated, info = env.step(
                np.zeros(env.action_space.shape[0])
            )  # step with zero action to get the first observation
            goal_img = info["target_obs"]  # [C, H, W] uint8 tensor

        combined = torch.stack([obs, goal_img], dim=0)
        show_images(
            combined,
            nrow=2,  # Both images in one row
            titles=["Init", "Goal"],
            save_path=f"{ep_folder}/state.pdf",
            close_fig=True,
            first_channel_only=False,
            clamp=False,
        )
        if waypoint_mode:
            # Aim MPPI at the nearest A* subgoal; `goal_img` stays the TRUE goal
            # for visualisation / success metric (unchanged below).
            waypoints = env.compute_waypoints(waypoint_spacing)
            wp_idx = 0
            wp_pos = waypoints[wp_idx]
            agent.set_goal(
                env._render_dot_at(wp_pos).to(dtype=torch.float32), wp_pos
            )
        else:
            agent.set_goal(
                goal_img.detach().clone().to(dtype=torch.float32),
                info["target_position"],
            )

        if ep == 0 and task_spec.get("diagnose", False):
            _diagnose_world_model(agent, env, obs)

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(
            desc="executing agent",
            total=steps_left,
            leave=True,
            disable=plan_cfg.logging.tqdm_silent,
        )
        t0 = True

        observations = [obs]
        infos = [info]

        prev_losses = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []

        cur_prior = None      # A* cardinal toward current waypoint (× cell_size)
        stall_count = 0       # consecutive steps with no GEODESIC progress
        best_dist = float("inf")  # best (smallest) geodesic dist-to-goal so far
        escape_uses = 0       # how many steps used the A* fallback

        while steps_left > 0:
            # while (not done and steps_left > 0):
            if waypoint_mode:
                # Advance to the next subgoal once the current one is reached,
                # re-encoding the goal only when the target actually changes.
                advanced = False
                while wp_idx < len(waypoints) - 1 and (
                    torch.norm(env.dot_position - waypoints[wp_idx]) < waypoint_reach
                ):
                    wp_idx += 1
                    advanced = True
                if advanced:
                    wp_pos = waypoints[wp_idx]
                    agent.set_goal(
                        env._render_dot_at(wp_pos).to(dtype=torch.float32), wp_pos
                    )
                if waypoint_action_prior or stall_escape:
                    # Cardinal direction (× cell_size) from current pos to the
                    # current waypoint = the A* move. Used to warm-start MPPI's
                    # mean (prior) and/or as the fallback when MPC stalls.
                    delta = (waypoints[wp_idx] - env.dot_position)
                    dr, dc = float(delta[0]), float(delta[1])
                    if abs(dr) >= abs(dc):
                        prior = [float(np.sign(dr)) * env.cell_size, 0.0]
                    else:
                        prior = [0.0, float(np.sign(dc)) * env.cell_size]
                    cur_prior = np.array(prior, dtype=np.float32)
                    if waypoint_action_prior:
                        agent.planner.action_prior = torch.tensor(
                            prior, device=agent.device, dtype=torch.float32
                        )
            plan_vis_path = (
                f"{ep_plan_vis_dir}/step{env.n_allowed_steps - steps_left}"
                if agent.decode_each_iteration
                else None
            )
            # first loop iter: obs is from reset(), then it is from step()
            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
            cell_before = env.agent_cell.copy() if hasattr(env, "agent_cell") else None
            astar_act = (
                env.astar_action_from_current()
                if (stall_escape and stall_count >= stall_patience)
                else None
            )
            if astar_act is not None:
                # MPC is stuck — recompute A* from the CURRENT cell and take its
                # first move (robust wherever the agent wandered to).
                action = astar_act.reshape(1, -1)  # [1, A]
                escape_uses += 1
            else:
                with torch.no_grad():
                    action = (
                        agent.act(
                            obs_tensor,
                            steps_left=steps_left,
                            t0=t0,
                            plan_vis_path=plan_vis_path,
                        )
                        .cpu()
                        .numpy()
                    )  # T, A
                if agent._prev_losses is not None:
                    prev_losses.append(agent._prev_losses)
                    prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
                    prev_elite_losses_std.append(agent._prev_elite_losses_std)
            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                observations.append(obs)
                infos.append(info)
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(
                    info["target_position"], info["dot_position"]
                )
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            # Progress = geodesic distance-to-goal strictly decreased. Catches
            # both stalls (no move) and oscillations (MPC moving the agent the
            # wrong way) → A* fallback triggers on any lack of real progress.
            if state_dist < best_dist - 1e-6:
                best_dist = state_dist
                stall_count = 0
            else:
                stall_count += 1
            pbar.set_postfix(
                {"success": success, "state_dist": state_dist, "esc": escape_uses}
            )
            if stop_on_success and (success or done):
                break
        pbar.close()
        if stall_escape:
            logger.info(
                f"ep {ep}: A* fallback used {escape_uses}× "
                f"({'succ' if success else 'fail'}, dist={state_dist:.0f})"
            )

        episode_observations.append(torch.stack(observations))
        episode_infos.append(infos)
        successes.append(success)
        distances.append(state_dist)

        save_path = f"{ep_folder}/agent_steps_{'succ' if success else 'fail'}.gif"
        if plan_cfg.logging.get("optional_plots", True):
            analyze_distances(
                episode_observations[-1],
                episode_infos[-1],
                str(ep_folder / "agent"),
                goal_position=agent.goal_position,
                goal_state=agent.goal_state,
                normalizer=agent.normalizer,
                model=agent.model,
                objective=agent.objective,
                device=agent.device,
            )
            plot_losses(
                prev_losses,
                prev_elite_losses_mean,
                prev_elite_losses_std,
                work_dir=ep_folder,
                num_act_stepped=agent.num_act_stepped,
            )
        if plan_cfg.logging.get("save_gif", True):
            save_gif(
                episode_observations[-1],
                save_path=save_path,
                show_frame_numbers=True,
                fps=20,
                init_frame=observations[0],
                goal_frame=goal_img,
            )
            logger.info(f"GIF saved to {save_path}")
        episode_end_time = time.time()  # Add this line
        episode_times.append(episode_end_time - episode_start_time)
    avg_episode_time = np.mean(episode_times)
    task_data = {
        "success_rate": np.mean(successes),
        "mean_state_dist": np.mean(distances),
        "avg_episode_time": avg_episode_time,
    }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data


def _main_eval_parallel(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes,
    n_parallel,
    prober=None,
):
    """Run eval with up to n_parallel episodes in lockstep, batching MPPI across envs.

    At each env step, instead of running K separate MPPI calls of batch=num_samples,
    we run ONE MPPI call of batch=K*num_samples. This amortises GPU kernel launch
    overhead and improves utilisation when num_samples alone under-saturates the GPU.
    """
    device = next(model.parameters()).device
    K = min(n_parallel, num_episodes)
    envs = [env_creator() for _ in range(K)]
    normalizer = envs[0].normalizer

    p = plan_cfg.planner
    num_samples = p.num_samples
    n_iters = p.n_iters
    num_elites = p.num_elites
    temperature = p.get("temperature", 0.005)
    max_std = p.get("max_std", 2.0)
    # NOTE: MPPIPlanner.plan() does NOT apply max_norms clipping (only CEMPlanner does).
    # To match sequential MPPI behavior, we do not clip actions here either.
    num_act_stepped = p.get("num_act_stepped", 1)
    sum_all_diffs = p.planning_objective.get("sum_all_diffs", True)
    save_gif_flag = plan_cfg.logging.get("save_gif", True)

    logger.info(
        f"Parallel eval: {num_episodes} episodes, K={K}, batch={K * num_samples} per MPPI iter"
    )

    all_successes, all_distances, all_times = [], [], []

    for batch_start in range(0, num_episodes, K):
        batch_K = min(K, num_episodes - batch_start)
        active_envs = envs[:batch_K]
        t_batch_start = time.time()

        # Reset envs and encode goals
        obs_list, goal_imgs, goal_encs = [], [], []
        for env in active_envs:
            obs, info = env.reset()
            obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
            goal_img = info["target_obs"].to(dtype=torch.float32)
            goal_enc = model.encode(
                normalizer.normalize_state(goal_img.to(device)).unsqueeze(0).unsqueeze(2)
            )  # (1, D, 1, H', W')
            obs_list.append(obs)
            goal_imgs.append(goal_img)
            goal_encs.append(goal_enc)

        goal_enc_batch = torch.cat(goal_encs, dim=0)  # (batch_K, D, 1, H', W')

        all_obs_k = [[obs_list[k]] for k in range(batch_K)]
        success_k = [False] * batch_K
        dist_k = [0.0] * batch_K
        n_steps = active_envs[0].n_allowed_steps

        for step_idx in range(n_steps):
            steps_left = n_steps - step_idx
            plan_length = min(p.plan_length, steps_left)
            action_dim = 2

            obs_tensors = [
                normalizer.normalize_state(
                    obs_list[k].to(dtype=torch.float32, device=device)
                ).unsqueeze(0).unsqueeze(2)
                for k in range(batch_K)
            ]
            obs_batch = torch.cat(obs_tensors, dim=0)  # (batch_K, C, 1, H, W)
            obs_expanded = obs_batch.repeat_interleave(num_samples, dim=0)

            means = torch.zeros(batch_K, plan_length, action_dim, device=device)
            stds = max_std * torch.ones(batch_K, plan_length, action_dim, device=device)

            for _ in range(n_iters):
                noise = torch.randn(batch_K, plan_length, num_samples, action_dim, device=device)
                actions_k = means.unsqueeze(2) + stds.unsqueeze(2) * noise
                actions_flat = rearrange(actions_k, "k t s a -> (k s) a t")

                with torch.no_grad():
                    predicted_states, _ = model.unroll(
                        obs_expanded,
                        actions_flat,
                        nsteps=plan_length,
                        unroll_mode="autoregressive",
                        ctxt_window_time=plan_cfg.get("ctxt_window_time", 1),
                        compute_loss=False,
                        return_all_steps=False,
                    )

                goal_expanded = goal_enc_batch[:batch_K].repeat_interleave(num_samples, dim=0)
                if goal_expanded.shape[2] != predicted_states.shape[2]:
                    goal_expanded = goal_expanded.expand(
                        -1, -1, predicted_states.shape[2], -1, -1
                    )
                diff = torch.nn.functional.mse_loss(
                    goal_expanded, predicted_states, reduction="none"
                ).mean(dim=(1, 3, 4))  # (batch_K*S, T)
                costs_flat = diff.sum(dim=1) if sum_all_diffs else diff[:, -1]
                costs = costs_flat.reshape(batch_K, num_samples)

                elite_idx = torch.topk(-costs, num_elites, dim=1).indices
                elite_costs = costs.gather(1, elite_idx)
                min_costs = costs.min(dim=1, keepdim=True).values
                scores = torch.exp(temperature * (min_costs - elite_costs))
                scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-9)

                idx_e = elite_idx[:, None, :, None].expand(
                    batch_K, plan_length, num_elites, action_dim
                )
                elite_acts = actions_k.gather(2, idx_e)

                s = scores[:, None, :, None]
                score_sum = scores.sum(dim=1).view(batch_K, 1, 1)
                means = (s * elite_acts).sum(dim=2) / (score_sum + 1e-9)
                stds = torch.sqrt(
                    (s * (elite_acts - means.unsqueeze(2)) ** 2).sum(dim=2)
                    / (score_sum + 1e-9)
                )

            # Stochastic elite selection — matches MPPIPlanner.plan() (eval_mode=False)
            chosen = torch.multinomial(scores, num_samples=1, replacement=True).squeeze(1)
            idx_c = chosen[:, None, None, None].expand(batch_K, plan_length, 1, action_dim)
            selected_acts = elite_acts.gather(2, idx_c).squeeze(2)  # (batch_K, T, A)
            post_noise = torch.randn(batch_K, action_dim, device=device)
            selected_acts = selected_acts + stds * post_noise.unsqueeze(1)

            actions_to_step = selected_acts[:, :num_act_stepped, :].cpu().numpy()
            for k in range(batch_K):
                for a in actions_to_step[k]:
                    obs_k, _, _, _, info_k = active_envs[k].step(a)
                all_obs_k[k].append(obs_k)
                obs_list[k] = obs_k
                result = active_envs[k].eval_state(
                    info_k["target_position"], info_k["dot_position"]
                )
                success_k[k] = result["success"]
                dist_k[k] = result["state_dist"]

        t_batch_end = time.time()
        per_ep_time = (t_batch_end - t_batch_start) / batch_K

        for k in range(batch_K):
            ep_idx = batch_start + k
            all_successes.append(success_k[k])
            all_distances.append(dist_k[k])
            all_times.append(per_ep_time)
            if save_gif_flag:
                ep_folder = eval_folder / f"ep_{ep_idx}"
                os.makedirs(ep_folder, exist_ok=True)
                label = "succ" if success_k[k] else "fail"
                obs_stack = torch.stack(all_obs_k[k])
                save_gif(
                    obs_stack,
                    save_path=str(ep_folder / f"agent_steps_{label}.gif"),
                    show_frame_numbers=True,
                    fps=20,
                    init_frame=all_obs_k[k][0],
                    goal_frame=goal_imgs[k],
                )

        logger.info(
            f"Batch {batch_start // K + 1}: success={np.mean(success_k[:batch_K]):.2f} "
            f"dist={np.mean(dist_k[:batch_K]):.4f} time={per_ep_time:.1f}s/ep"
        )

    task_data = {
        "success_rate": float(np.mean(all_successes)),
        "mean_state_dist": float(np.mean(all_distances)),
        "avg_episode_time": float(np.mean(all_times)),
    }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data


### Goal-conditioned agent for planning ###
class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None,
        normalizer: Optional[Callable] = None,
        loc_prober: Optional[Callable] = None,
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None,
        value_head: Optional[Callable] = None,
    ):
        self.plan_cfg = plan_cfg
        self.env = env
        self.model = model
        self.device = next(model.parameters()).device
        self.loc_prober = loc_prober
        self.img_prober = img_prober
        self.value_head = value_head
        self.normalizer = normalizer

        # Action snapping: the maze world-model is trained ONLY on cardinal
        # actions of magnitude cell_size ({(±c,0),(0,±c)}); MPPI samples
        # continuous Gaussian actions on both axes → out-of-distribution → the
        # model predicts garbage → the agent never moves. Snapping each action
        # to the nearest cardinal × cell_size (exactly what env.step does) feeds
        # the model in-distribution actions. cell_size from the env.
        self.snap_actions = bool(
            plan_cfg.planner.get("snap_actions_to_grid", False)
        ) if plan_cfg is not None else False
        self.cell_size = float(getattr(env, "cell_size", 1) or 1)

        # Set default values if plan_cfg is None
        if plan_cfg is None:
            self.decode_each_iteration = False
            self.num_act_stepped = 1
            self.planner = None
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_name = plan_cfg.planner.get("planner_name", "cem")
            planner_class_name = planner_name_map[planner_name]
            planner_class = globals()[planner_class_name]
            if planner_class is not None:
                self.planner = planner_class(
                    unroll=self.unroll,
                    action_dim=action_dim,
                    decode_loc_to_pixel=self.decode_loc_to_pixel,
                    **plan_cfg.planner,
                )
            else:
                logger.info("No planner provided in GCAgent.")
                self.planner = None

        self.goal_state = None
        self.goal_position = None
        self.goal_state_enc = None
        self._prev_losses = None

    def set_goal(self, goal_state, goal_position=None):
        self.goal_position = goal_position
        self.goal_state = goal_state
        # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
        self.goal_state_enc = self.model.encode(
            self.normalizer.normalize_state(goal_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )
        objective_name = self.plan_cfg.planner.planning_objective.get(
            "objective_type", "repr_target_dist"
        )
        objective_class_name = objective_name_map[objective_name]
        objective_class = globals()[objective_class_name]
        self.objective = objective_class(
            target_enc=self.goal_state_enc,
            target_position=self.goal_position,
            prober=self.loc_prober,
            value_head=self.value_head,
            normalizer=self.normalizer,
            env=self.env,
            **self.plan_cfg.planner.planning_objective,
        )
        self.planner.set_objective(self.objective)

    def unroll(self, obs_init, actions, repeat_batch=True):
        """
        Unroll the model for planning.

        Args:
            obs_init: [B, C, T, H, W]
            actions: [B, A, T]

        Returns:
            predicted_states: [B, D, T, H, W]
        """
        batch_size = actions.shape[0]
        nsteps = actions.shape[2]
        if self.snap_actions:
            actions = self._snap_to_grid(actions)
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1,
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted_states

    def _snap_to_grid(self, actions):
        """Snap continuous actions to the nearest cardinal × cell_size.

        actions: [B, A=2, T]. Mirrors ``MazeEnv.step``'s discretisation (move
        along the max-magnitude axis, sign-preserving, magnitude cell_size) so
        the world-model receives the same in-distribution action it was trained
        on. The continuous samples still drive MPPI's mean/std updates; only the
        model-facing copy is snapped.
        """
        a_r = actions[:, 0, :]
        a_c = actions[:, 1, :]
        dom_r = a_r.abs() >= a_c.abs()
        snapped = torch.zeros_like(actions)
        snapped[:, 0, :] = torch.where(
            dom_r, torch.sign(a_r), torch.zeros_like(a_r)
        ) * self.cell_size
        snapped[:, 1, :] = torch.where(
            ~dom_r, torch.sign(a_c), torch.zeros_like(a_c)
        ) * self.cell_size
        return snapped

    def decode_loc_to_pixel(self, predicted_encs, wall_x=None, door_y=None):
        """
        Decode the predicted encodings into frames.

        Args:
            predicted_encs: [B, D, T, H, W]

        Returns:
            np.array of shape [B, T, H, W, C] on cpu for visualization.
        """
        assert self.loc_prober is not None
        B, D, T, H, W = predicted_encs.shape
        out = self.loc_prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()  # B T 2
        out = self.normalizer.unnormalize_location(out)  # B T 2
        frames = self.env.coord_to_pixel(out, wall_x=wall_x, door_y=door_y)  # B T C H W
        frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # B T H W C
        return frames

    def act(self, obs, steps_left=None, t0=False, plan_vis_path=None):
        planning_result = self.planner.plan(
            obs,
            steps_left=steps_left,
            eval_mode=True,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )
        self._prev_losses = planning_result.losses
        self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
        self._prev_elite_losses_std = planning_result.prev_elite_losses_std
        return planning_result.actions[: self.num_act_stepped]  # T, A


### Planning objectives to minimize ###
class ReprTargetDistMPCObjective:
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: [B, D, T, H, W]
            keepdims: if True, return [B, T], else return [B]

        Returns:
            diff: [B, T] else [B] if sum_all_diffs or not keepdims
        """
        if self.sum_all_diffs:
            keepdims = True
        target = self.target_enc
        if target.shape != encodings.shape:
            target = target.expand(encodings.shape[0], -1, encodings.shape[2], -1, -1)

        metric = torch.nn.MSELoss(reduction="none")
        diff = metric(target, encodings).mean(dim=(1, 3, 4))  # B T
        if not keepdims:
            diff = diff[:, -1]
        if self.sum_all_diffs:
            diff = diff.sum(dim=1)
        return diff


class ReprDistCollisionMPCObjective(ReprTargetDistMPCObjective):
    """``repr_dist`` plus a penalty for predicted states that land on walls.

    The greedy ``repr_dist`` objective drives the agent straight at the goal in
    latent space → into walls (local minima, the maze 0% blocker). Here we keep
    that term but add a non-greedy one: decode each predicted latent to a pixel
    position via the location prober, map it to a maze cell, and penalise
    trajectories whose predicted states fall on *wall* cells. MPPI is then
    steered toward action sequences that stay on the path while still heading
    to the goal.

    Needs the location ``prober`` (to decode latents → xy) and the ``env`` (for
    the current episode's wall mask + cell geometry); both are passed by
    ``GCAgent.set_goal``. The wall mask is captured at construction, so the
    objective must be (re)built per episode after ``env.reset`` — which is what
    ``set_goal`` does.
    """

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        prober=None,
        env=None,
        collision_coeff: float = 0.3,
        **kwargs,
    ):
        super().__init__(target_enc, sum_all_diffs=sum_all_diffs, **kwargs)
        assert prober is not None and env is not None, (
            "repr_dist_collision objective requires a loc prober and the env"
        )
        self.prober = prober
        self.normalizer = env.normalizer
        self.cell_size = env.cell_size
        self.maze_h = env.maze_height
        self.maze_w = env.maze_width
        # 1.0 = path, 0.0 = wall ; (H_cell, W_cell) float, current episode's maze
        self.path_grid = env.maze_grid.detach().to(torch.float32)
        self.collision_coeff = collision_coeff

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        base = super().__call__(encodings, keepdims=keepdims)

        # prober MLP is float32; cast defensively in case latents are bf16
        xy = self.prober.apply_head(encodings.float())  # [B, 2, T] (normalized)
        xy = xy.permute(0, 2, 1)  # [B, T, 2]
        xy = self.normalizer.unnormalize_location(xy)  # [B, T, 2] pixel (row, col)

        offset = (self.cell_size - 1) / 2.0
        cells = torch.round((xy - offset) / self.cell_size)
        r = cells[..., 0].clamp(0, self.maze_h - 1).long()
        c = cells[..., 1].clamp(0, self.maze_w - 1).long()
        grid = self.path_grid.to(encodings.device)
        is_path = grid[r, c]  # [B, T] — 1 on path, 0 on wall
        pen = 1.0 - is_path  # [B, T]

        # Mirror the base reduction so shapes match (see ReprTargetDistMPCObjective)
        if self.sum_all_diffs:
            pen = pen.sum(dim=1)
        elif not keepdims:
            pen = pen[:, -1]

        return base + self.collision_coeff * pen


class ProbePositionMPCObjective:
    """Plan in POSITION space via the location probe, not full-latent distance.

    Root issue with ``repr_dist`` in the maze: the 2-channel obs (dot + static
    wall mask) yields a latent dominated by the walls (constant within an
    episode); the tiny dot's position contributes little, so latent
    distance-to-goal is nearly flat → MPPI gets almost no signal → the agent
    barely moves. Here we decode each predicted latent to a pixel position with
    the probe and minimise the distance to the goal *position* directly — a
    clean, wall-invariant signal. ``target_position`` is the goal (or current
    waypoint) in pixel space, supplied by ``GCAgent.set_goal``.
    """

    def __init__(
        self,
        target_enc=None,
        target_position=None,
        prober=None,
        normalizer=None,
        sum_all_diffs: bool = True,
        **kwargs,
    ):
        assert prober is not None and target_position is not None, (
            "probe_pos objective needs a loc prober and a target_position"
        )
        self.prober = prober
        self.normalizer = normalizer
        self.sum_all_diffs = sum_all_diffs
        tp = target_position
        if isinstance(tp, torch.Tensor):
            tp = tp.detach().float()
        else:
            tp = torch.tensor(tp, dtype=torch.float32)
        self.target_position = tp  # (2,) pixel (row, col)

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        xy = self.prober.apply_head(encodings.float()).permute(0, 2, 1)  # [B, T, 2]
        xy = self.normalizer.unnormalize_location(xy)  # pixel (row, col)
        tgt = self.target_position.to(xy.device)
        d = ((xy - tgt) ** 2).mean(dim=-1)  # [B, T]
        if self.sum_all_diffs:
            return d.sum(dim=1)
        if not keepdims:
            return d[:, -1]
        return d


class LearnedValueMPCObjective:
    """Plan by MAXIMISING a learned goal-conditioned value, TD-MPC style.

    Root issue with ``repr_dist``/``probe_pos``: both are hand-crafted geometric
    costs (latent MSE or decoded-position distance). Latent MSE is dominated by
    the static wall mask; even position distance is a crude as-the-crow-flies
    proxy that ignores walls between agent and goal. Here the cost is the negative
    of a *learned* value function ``V(z, z_goal)`` (Hansen et al., TD-MPC 2022/2024)
    trained on the world model's own rollouts to predict the discounted
    return-to-goal (≈ ``gamma ** steps_to_goal``). MPPI then minimises ``-V``, i.e.
    maximises value, optimising a quantity that correlates with *task success*
    (true steps-to-goal, walls included) rather than representation distance.

    The ``value_head`` and the goal latent (``target_enc``) are supplied by
    ``GCAgent.set_goal``. An optional small ``probe_pos`` blend (``blend_coeff``)
    can stabilise early planning, but defaults to 0 (pure learned value).
    """

    def __init__(
        self,
        target_enc=None,
        value_head=None,
        gamma: float = 0.95,
        sum_all_diffs: bool = True,
        blend_coeff: float = 0.0,
        prober=None,
        normalizer=None,
        target_position=None,
        **kwargs,
    ):
        assert value_head is not None and target_enc is not None, (
            "learned_value objective needs a trained value_head and a goal latent"
        )
        self.value_head = value_head
        self.goal_enc = target_enc  # [1, C, 1, h, w]
        self.gamma = float(gamma)
        self.sum_all_diffs = sum_all_diffs
        # optional position-distance blend for early-training stabilisation
        self.blend_coeff = float(blend_coeff)
        self._pos_obj = None
        if self.blend_coeff > 0 and prober is not None and target_position is not None:
            self._pos_obj = ProbePositionMPCObjective(
                target_position=target_position, prober=prober,
                normalizer=normalizer, sum_all_diffs=sum_all_diffs,
            )

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        # encodings: [B, C, T, h, w]
        B, C, T, H, W = encodings.shape
        v = self.value_head(encodings.float(), self.goal_enc.float())  # [B, T] in (0,1)
        cost = 1.0 - v  # minimise -> maximise value (cost-to-go in [0,1])

        if self.sum_all_diffs:
            disc = self.gamma ** torch.arange(T, device=encodings.device, dtype=cost.dtype)
            out = (cost * disc.view(1, T)).sum(dim=1)  # [B]
        elif keepdims:
            out = cost  # [B, T]
        else:
            out = cost[:, -1]  # [B]

        if self._pos_obj is not None:
            out = out + self.blend_coeff * self._pos_obj(encodings, keepdims=keepdims)
        return out


### Planning optimizers interface ###
class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None


class Planner(ABC):
    def __init__(self, unroll: Callable, **kwargs):
        self.unroll = unroll
        self.objective = None

    def set_objective(self, objective: Callable):
        self.objective = objective

    @abstractmethod
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
    ):
        pass

    def cost_function(
        self, actions: torch.Tensor, obs_init: torch.Tensor
    ) -> torch.Tensor:
        predicted_encs = self.unroll(obs_init, actions)
        return self.objective(predicted_encs)


### Specific planning optimizers ###
class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 30,
        num_samples: int = 300,
        plan_length: int = 15,
        action_dim: int = 2,
        var_scale: float = 1,
        num_elites: int = 10,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = True,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel

    @torch.no_grad()
    def plan(
        self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None
    ):
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        # Initialize mean and std for the action distribution
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(
            plan_length, self.action_dim, device=self.device
        )

        # Initialize actions tensor
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # CEM iterations
        for _ in range(self.n_iters):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A

            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm = self.max_norms[0]
                eps = 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                # calculate min and max allowed step sizes
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.ones_like(norms) * 0
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (
                    norms + eps
                )
                actions = actions * coeff

            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)

        # Return the first action(s)
        a = mean

        return PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = False,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel
        self._prev_mean = None
        # Optional warm-start: a per-step action prior (e.g. the A* cardinal
        # direction toward the current waypoint). Centers MPPI's search on the
        # right move so probe-decoding noise can't flip the direction; the
        # world-model + objective still refine/override via the cost. [A,] tensor.
        self.action_prior = None

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ):
        """
        Args:
                obs_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        if self.action_prior is not None:
            mean = self.action_prior.to(self.device).view(1, self.action_dim).expand(
                plan_length, self.action_dim
            ).clone()
        else:
            mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(
            plan_length, self.action_dim, device=self.device
        )
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # MPPI iterations
        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A
            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (  # T B A
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)
        # Select action
        score = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]  # T, A
        self._prev_mean = mean
        if not eval_mode:
            actions += std * torch.randn(
                self.action_dim, device=std.device, generator=self.local_generator
            )

        return PlanningResult(
            actions=actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
