import gym
import d4rl
from tqdm import tqdm

from model import *
from utils import *
#from train_utils import *
import warnings


def main(args: ArgStorage) -> None:
    env = gym.make(args.env_name)
    print('Environmnet created')
    state_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    prompt_dim = state_dim // 2

    model = Net(state_dim, ac_dim, prompt_dim)
    model.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))

    state = torch.tensor(env.reset(), dtype=torch.float32)
    prompt = torch.zeros(prompt_dim, dtype=torch.float32)
    # [3, 2], [1, 4], [4, 4], [2, 1]
    prompt[0] = 1
    prompt[1] = 4
    # prompt = torch.tensor(np.zeros([3, 2]), dtype=torch.float32)
    for i in range(10000):
        # env.render()
        with torch.no_grad():
            action = model(state, prompt).detach().numpy().squeeze()
        state, _, _, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        # print(state)
        # print(action)
        done = (state[:2] - prompt).abs().sum() < 0.2
        if done:
            print('Goal reached in {} steps'.format(i))
            break

if __name__ == '__main__':
    args = ArgStorage(
        seed=2024,
        env_name='maze2d-medium-v1',
    )
    
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main(args)
