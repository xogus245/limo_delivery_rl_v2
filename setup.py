from glob import glob

from setuptools import find_packages, setup

package_name = "limo_delivery_rl_v2"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test", "test.*", "legacy_v1", "legacy_v1.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "gymnasium", "numpy", "stable-baselines3"],
    zip_safe=True,
    maintainer="kth",
    maintainer_email="kth@todo.todo",
    description=(
        "Waypoint-following RL environment for LIMO: Nav2 plans the global path, "
        "a PPO policy outputs (v, omega) directly."
    ),
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "delivery_v2_smoke = limo_delivery_rl_v2.delivery_env:main",
            "delivery_v2_train_ppo = limo_delivery_rl_v2.train_ppo:main",
            "delivery_v2_eval_ppo = limo_delivery_rl_v2.evaluate_ppo:main",
        ],
    },
)
