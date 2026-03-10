import optuna
from stable_baselines3 import PPO
from src.PersonFollower.quadx_simple_follow_env import QuadXSimpleFollowEnv


def objective(trial):
    params = trial.params

    env = QuadXSimpleFollowEnv(
        num_persons=5,
        size=10.0,  # map size
        flight_mode=1,
        drone_spawn_mode="random",
        render_mode=None  # Set to "human" if you want to see the 3D drone
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=0.0003
    )

    model.learn(N)
    avg_dist = evaluate(model)
    distance_history = []
    return avg_dist

# si el objetive devuleve la distancia, hay que minimizar, si devuelve el reward maximizar
study = optuna.create_study(direction= "minimize") # trials anteriores
assert len(study.trials) == 0


study.optimize(objective, n_trials=3)
assert len(study.trials) == 4

Btrial = study.best_trial
print("Best trial:", Btrial)


