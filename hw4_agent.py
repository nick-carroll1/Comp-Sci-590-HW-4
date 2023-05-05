import argparse
import logging
import timeit

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

import gymnasium as gym

import hw4_utils as utils


torch.set_num_threads(4)


logging.basicConfig(
    format=("[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO
)


class PolicyNetwork(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.gelu(
            self.conv3(  # bsz*T x hidden_dim x H3 x W3
                F.gelu(
                    self.conv2(
                        F.gelu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))
                    )
                )
            )
        )

        # flatten with MLP
        Z = F.gelu(self.fc1(Z.view(bsz * T, -1)))  # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)

        return self.fc2(Z), X, prev_state

    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        logits, prev_state, prev_prev_state = self(x, prev_state)
        # take highest scoring action
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state


class MyPolicyNetwork(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960 * 2, 2 * args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim * 2, naction)

    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.gelu(
            self.conv3(  # bsz*T x hidden_dim x H3 x W3
                F.gelu(
                    self.conv2(
                        F.gelu(self.conv1(X.reshape(-1, self.iC, self.iH, self.iW)))
                    )
                )
            )
        )

        if prev_state is not None:
            Z2 = F.gelu(
                self.conv3(  # bsz*T x hidden_dim x H3 x W3
                    F.gelu(
                        self.conv2(
                            F.gelu(self.conv1(X.reshape(-1, self.iC, self.iH, self.iW)))
                        )
                    )
                )
            )

        else:
            Z2 = torch.zeros(Z.shape)

        # flatten with MLP
        Z = F.gelu(
            self.fc1(torch.cat((Z.view(bsz * T, -1), Z2.view(bsz * T, -1)), axis=1))
        )  # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)

        return self.fc2(Z), X, prev_state

    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        logits, prev_state, prev_prev_state = self(x, prev_state)
        # take highest scoring action
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state


def pg_step(
    stepidx,
    model,
    optimizer,
    scheduler,
    envs,
    observations,
    prev_state,
    prev_lives,
    args,
    bsz=4,
    target=None,
    replay=None,
    epsilon=1,
):
    if envs is None:
        envs = [gym.make(args.env) for _ in range(bsz)]
        observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
        observations = torch.stack(  # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
            [utils.preprocess_observation(obs) for obs in observations]
        ).unsqueeze(1)
        prev_state = None

    logits, rewards, actions = [], [], []
    if args.mode == "train":
        observationList, prev_stateList, prev_prev_stateList = [], [], []
        prev_prev_state = None
    not_terminated = torch.ones(bsz)  # agent is still alive

    if target is not None:
        target_logits = []

    for t in range(args.unroll_length):
        if args.mode == "train":
            observationList.append(observations)
            prev_stateList.append(prev_state)
            prev_prev_stateList.append(prev_prev_state)
        logits_t, prev_state, prev_prev_state = model(
            observations, prev_state
        )  # logits are bsz x 1 x naction
        logits.append(logits_t)
        # sample actions
        if torch.rand(1) < 10_000 / (epsilon + t):
            actions_t = torch.randint(0, envs[0].action_space.n, (bsz,))
        else:
            actions_t = Categorical(logits=logits_t.squeeze(1)).sample()
        actions.append(actions_t.view(-1, 1))  # bsz x 1
        # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
        # keep track of when deaths occur for each env
        lives = torch.tensor([env.ale.lives() for env in envs])
        deaths = prev_lives - lives
        prev_lives = lives
        rewards_t = torch.tensor([eo[1] for eo in env_outputs])
        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor(
            [env.ale.lives() == args.start_nlives for env in envs]
        )
        not_terminated.mul_(still_alive.float())
        if args.mode == "baseline":
            rewards.append(rewards_t * not_terminated)
        if args.mode == "train":
            rewards.append(rewards_t * (1 - deaths) - 50 * deaths)
        observations = torch.stack(
            [utils.preprocess_observation(eo[0]) for eo in env_outputs]
        ).unsqueeze(1)

    if args.mode == "baseline":
        # calculate reward-to-go
        r2g = torch.zeros(bsz, args.unroll_length)
        curr_r = 0
        for r in range(args.unroll_length - 1, -1, -1):
            curr_r = rewards[r] + args.discounting * curr_r
            r2g[:, r].copy_(curr_r)

        adv = (r2g - r2g.mean()) / (r2g.std() + 1e-7)
        logits = torch.cat(logits, dim=1)  # bsz x T x naction
        actions = torch.cat(actions, dim=1)  # bsz x T
        cross_entropy = F.cross_entropy(
            logits.view(-1, logits.size(2)), actions.view(-1), reduction="none"
        )
        pg_loss = (cross_entropy.view_as(actions) * adv).mean()

    with torch.no_grad():
        if args.mode == "train":
            # stack rewards
            while prev_stateList[0] is None or prev_prev_stateList[0] is None:
                prev_stateList.pop(0)
                prev_prev_stateList.pop(0)
                observationList.pop(0)
                rewards.pop(0)
                actions.pop(0)
                logits.pop(0)
            observationStack = torch.stack(observationList[1:]).transpose(
                0, 1
            )  # bsz x T x 1 x ic x iH x iW
            prev_stateStack = torch.stack(prev_stateList[1:]).transpose(
                0, 1
            )  # bsz x T x 1 x ic x iH x iW
            prev_prev_stateStack = torch.stack(prev_prev_stateList[1:]).transpose(
                0, 1
            )  # bsz x T x 1 x ic x iH x iW
            del observationList, prev_stateList, prev_prev_stateList
            prev_rewards = torch.stack(rewards[:-1]).transpose(0, 1)  # bsz x T
            prev_actions = (
                torch.stack(actions[:-1]).squeeze().transpose(0, 1)
            )  # bsz x T
            if replay is not None:
                if replay[0].size(1) >= 200:
                    mask = torch.randint(high=replay[0].size(1), size=(120,))
                    replay = (
                        replay[0][:, mask],
                        replay[1][:, mask],
                        replay[2][:, mask],
                        replay[3][:, mask],
                        replay[4][:, mask],
                    )
                replay = (
                    torch.cat((replay[0], observationStack), axis=1),
                    torch.cat((replay[1], prev_stateStack), axis=1),
                    torch.cat((replay[2], prev_prev_stateStack), axis=1),
                    torch.cat((replay[3], prev_rewards), axis=1),
                    torch.cat((replay[4], prev_actions), axis=1),
                )
            else:
                replay = (
                    observationStack,
                    prev_stateStack,
                    prev_prev_stateStack,
                    prev_rewards,
                    prev_actions,
                )

            if replay[2].size(1) > 150:
                mask = torch.randint(high=replay[0].size(1), size=(150,))
                observationStack = replay[0][:, mask]
                prev_stateStack = replay[1][:, mask]
                prev_prev_stateStack = replay[2][:, mask]
                prev_rewards = replay[3][:, mask]
                prev_actions = replay[4][:, mask]

    if args.mode == "train":
        logits, garbage, more_garbage = model(
            prev_stateStack, prev_prev_stateStack
        )  # logits are bsz x T x naction

        with torch.no_grad():
            target_logits, garbage, more_garbage = target(
                observationStack, prev_stateStack
            )  # logits are bsz x T x naction
            del garbage, more_garbage

            prev_rewards = (prev_rewards - prev_rewards.mean()) / (
                prev_rewards.std() + 1e-7
            )  # bsz x T x naction

            y_target = (
                prev_rewards
                + args.discounting * torch.max(target_logits.detach(), dim=-1)[0]
            )  # bsz x T

        y_output = logits.gather(dim=-1, index=prev_actions.unsqueeze(-1)).squeeze(
            -1
        )  # bsz x T

        pg_loss = F.mse_loss(y_output, y_target)

    total_loss = pg_loss

    stats = {
        "mean_return": sum(r.mean() for r in rewards) / len(rewards),
        "pg_loss": pg_loss.item(),
    }
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
    optimizer.step()
    scheduler.step()

    # reset any environments that have ended
    for b in range(bsz):
        if not_terminated[b].item() == 0:
            obs = envs[b].reset(seed=stepidx + b)[0]
            observations[b].copy_(utils.preprocess_observation(obs))

    return stats, envs, observations, prev_state, prev_lives, replay, epsilon + t


def train(args):
    T = args.unroll_length
    B = args.batch_size

    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    if args.mode == "baseline":
        model = PolicyNetwork(naction, args)
    elif args.mode == "train":
        model = MyPolicyNetwork(naction, args)
        q_target = MyPolicyNetwork(naction, args)
        if args.load_path is not None:
            checkpoint = torch.load(args.load_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            q_target.load_state_dict(checkpoint["model_state_dict"])
        replay_buffer = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def lr_lambda(
        epoch,
    ):  # multiplies learning rate by value returned; can be used to decay lr
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def checkpoint():
        if args.save_path is None:
            return
        logging.info("Saving checkpoint to {}".format(args.save_path))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": args,
            },
            args.save_path,
        )

    timer = timeit.default_timer
    last_checkpoint_time = timer()
    envs, observations, prev_state, prev_lives = None, None, None, 3 * torch.ones(B)
    frame = 0
    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame
        if args.mode == "train":
            if frame == 0 or frame % (5 * T * B) == 0:
                q_target.load_state_dict(model.state_dict())
        epsilon = 1
        (
            stats,
            envs,
            observations,
            prev_state,
            prev_lives,
            replay_buffer,
            epsilon,
        ) = pg_step(
            frame,
            model,
            optimizer,
            scheduler,
            envs,
            observations,
            prev_state,
            prev_lives,
            args,
            bsz=B,
            target=q_target if args.mode == "train" else None,
            replay=replay_buffer if args.mode == "train" else None,
            epsilon=epsilon,
        )
        frame += T * B  # here steps means number of observations
        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        sps = (frame - start_frame) / (timer() - start_time)
        logging.info(
            "Frame {:d} @ {:.1f} FPS: pg_loss {:.3f} | mean_ret {:.3f}".format(
                frame, sps, stats["pg_loss"], stats["mean_return"]
            )
        )

        if frame > 0 and frame % (args.eval_every * T * B) == 0:
            utils.validate(model, render=args.render)
            model.train()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--env", type=str, default="ALE/MsPacman-v5", help="gym environment"
)
parser.add_argument(
    "--mode",
    default="train",
    choices=[
        "train",
        "valid",
        "baseline",
    ],
    help="training or validation mode",
)
parser.add_argument(
    "--total_frames",
    default=1000000,
    type=int,
    help="total environment frames to train for",
)
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument(
    "--unroll_length", default=80, type=int, help="unroll length (time dimension)"
)
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument(
    "--discounting", default=0.99, type=float, help="discounting factor"
)
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument(
    "--grad_norm_clipping", default=10.0, type=float, help="Global gradient norm clip."
)
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument(
    "--min_to_save", default=5, type=int, help="save every this many minutes"
)
parser.add_argument(
    "--eval_every", default=50, type=int, help="eval every this many updates"
)
parser.add_argument(
    "--render", action="store_true", help="render game-play at validation time"
)


if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    if args.mode == "train" or args.mode == "baseline":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env
        model = MyPolicyNetwork(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args)
