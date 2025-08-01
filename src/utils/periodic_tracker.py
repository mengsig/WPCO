"""Utility for robust periodic task tracking in asynchronous environments."""

import threading
from typing import Dict, Callable, Optional, Any


class PeriodicTaskTracker:
    """
    Tracks periodic tasks and ensures they execute even if exact step counts are missed.

    This is particularly useful in asynchronous training where step counters might skip
    exact multiples due to race conditions.
    """

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def register_task(
        self,
        name: str,
        interval: int,
        callback: Callable,
        start_step: int = 0,
        enabled: bool = True,
    ):
        """
        Register a periodic task.

        Args:
            name: Unique identifier for the task
            interval: How often to run the task (in steps)
            callback: Function to call when task should run
            start_step: Step count to start from (default: 0)
            enabled: Whether the task is initially enabled
        """
        with self.lock:
            self.tasks[name] = {
                "interval": interval,
                "callback": callback,
                "last_executed": start_step,
                "next_target": start_step + interval,
                "enabled": enabled,
                "executions": 0,
            }

    def check_and_execute(self, current_step: int) -> Dict[str, bool]:
        """
        Check all tasks and execute those that are due.

        Args:
            current_step: Current step count

        Returns:
            Dictionary mapping task names to whether they were executed
        """
        executed = {}

        with self.lock:
            for name, task in self.tasks.items():
                if not task["enabled"]:
                    executed[name] = False
                    continue

                # Check if we've passed or reached the next target
                if current_step >= task["next_target"]:
                    try:
                        # Execute the callback
                        task["callback"](current_step)

                        # Update tracking
                        task["last_executed"] = current_step
                        task["executions"] += 1

                        # Calculate next target based on the interval
                        # This ensures we don't miss intervals even if steps are skipped
                        missed_intervals = (current_step - task["next_target"]) // task[
                            "interval"
                        ]
                        task["next_target"] += task["interval"] * (missed_intervals + 1)

                        executed[name] = True
                    except Exception as e:
                        print(f"Error executing periodic task '{name}': {e}")
                        executed[name] = False
                else:
                    executed[name] = False

        return executed

    def force_execute(self, name: str, current_step: int) -> bool:
        """Force execution of a specific task."""
        with self.lock:
            if name in self.tasks:
                try:
                    self.tasks[name]["callback"](current_step)
                    self.tasks[name]["last_executed"] = current_step
                    self.tasks[name]["executions"] += 1
                    return True
                except Exception as e:
                    print(f"Error force executing task '{name}': {e}")
                    return False
        return False

    def set_enabled(self, name: str, enabled: bool):
        """Enable or disable a task."""
        with self.lock:
            if name in self.tasks:
                self.tasks[name]["enabled"] = enabled

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks."""
        with self.lock:
            return {
                name: {
                    "interval": task["interval"],
                    "last_executed": task["last_executed"],
                    "next_target": task["next_target"],
                    "enabled": task["enabled"],
                    "executions": task["executions"],
                }
                for name, task in self.tasks.items()
            }

    def reset_task(self, name: str, current_step: int):
        """Reset a task's tracking."""
        with self.lock:
            if name in self.tasks:
                self.tasks[name]["last_executed"] = current_step
                self.tasks[name]["next_target"] = (
                    current_step + self.tasks[name]["interval"]
                )
                self.tasks[name]["executions"] = 0


class RobustPeriodicChecker:
    """
    A simpler alternative for checking if a periodic action should occur.
    Handles cases where exact step counts might be missed.
    """

    def __init__(self, interval: int, start_step: int = 0):
        self.interval = interval
        self.last_executed = start_step
        self.next_target = start_step + interval
        self.lock = threading.Lock()

    def should_execute(self, current_step: int) -> bool:
        """
        Check if the periodic action should execute at the current step.

        Returns True if the action should execute, False otherwise.
        """
        with self.lock:
            if current_step >= self.next_target:
                # Calculate how many intervals we might have missed
                missed_intervals = (current_step - self.next_target) // self.interval
                self.next_target += self.interval * (missed_intervals + 1)
                self.last_executed = current_step
                return True
            return False

    def reset(self, current_step: int):
        """Reset the checker."""
        with self.lock:
            self.last_executed = current_step
            self.next_target = current_step + self.interval

    def force_next_at(self, target_step: int):
        """Force the next execution to occur at a specific step."""
        with self.lock:
            self.next_target = target_step

