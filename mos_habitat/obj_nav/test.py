import habitat


def main():
    config = habitat.get_config(
        config_path="/home/sai/Desktop/multi-object-search/sem_objnav/sem_objnav/obj_nav/cfg/h_config.yaml"
    )

    with habitat.Env(config=config) as env:
        env.reset()
        for _ in range(100):
            obs = env.step(env.action_space.sample())
            print(obs)


if __name__ == "__main__":
    main()
