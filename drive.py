"""Interactive two-motor control for an L298N on a Raspberry Pi.

Wire the driver's ENA, IN1, IN2, ENB, IN3, and IN4 to the BCM pins below.
Their physical header pins (7, 11, 13, 15, 29, 31) are all in one column.
Remove the ENA/ENB jumpers on the L298N module so the Pi can control speed.
"""

import argparse
from contextlib import ExitStack

from gpiozero import OutputDevice, PWMOutputDevice


# BCM GPIO numbering; every signal is on the same physical header column.
# L298N ENA, IN1, IN2, ENB, IN3, IN4 -> physical pins 7, 11, 13, 15, 29, 31.
ENA_PIN = 4
IN1_PIN = 17
IN2_PIN = 27
ENB_PIN = 22
IN3_PIN = 5
IN4_PIN = 6


class L298N:
    def __init__(self, devices: ExitStack) -> None:
        self.ena = devices.enter_context(PWMOutputDevice(ENA_PIN, initial_value=0))
        self.in1 = devices.enter_context(OutputDevice(IN1_PIN, initial_value=False))
        self.in2 = devices.enter_context(OutputDevice(IN2_PIN, initial_value=False))
        self.enb = devices.enter_context(PWMOutputDevice(ENB_PIN, initial_value=0))
        self.in3 = devices.enter_context(OutputDevice(IN3_PIN, initial_value=False))
        self.in4 = devices.enter_context(OutputDevice(IN4_PIN, initial_value=False))

    @staticmethod
    def _set_motor(
        enable: PWMOutputDevice,
        input_a: OutputDevice,
        input_b: OutputDevice,
        speed: float,
    ) -> None:
        input_a.value = speed > 0
        input_b.value = speed < 0
        enable.value = abs(speed)

    def drive(self, left_speed: float, right_speed: float) -> None:
        self._set_motor(self.ena, self.in1, self.in2, left_speed)
        self._set_motor(self.enb, self.in3, self.in4, right_speed)

    def stop(self) -> None:
        self.drive(0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive two motors through an L298N.")
    parser.add_argument(
        "--speed",
        type=int,
        default=60,
        metavar="PERCENT",
        help="motor speed from 1 to 100 (default: 60)",
    )
    args = parser.parse_args()
    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")

    speed = args.speed / 100
    commands = {
        "w": (speed, speed, "forward"),
        "s": (-speed, -speed, "backward"),
        "a": (-speed, speed, "turn left"),
        "d": (speed, -speed, "turn right"),
        "x": (0, 0, "stop"),
    }

    print("Commands: w=forward, s=backward, a=left, d=right, x=stop, q=quit")
    with ExitStack() as devices:
        driver = L298N(devices)
        try:
            while True:
                try:
                    command = input("> ").strip().lower()
                except EOFError:
                    break
                if command == "q":
                    break
                if command not in commands:
                    print("Unknown command. Use w, s, a, d, x, or q.")
                    continue
                left, right, description = commands[command]
                driver.drive(left, right)
                print(description)
        except KeyboardInterrupt:
            print("\nStopping motors.")
        finally:
            driver.stop()


if __name__ == "__main__":
    main()
