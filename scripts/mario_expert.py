import json
import logging
import random

import cv2
from mario_environment import MarioEnvironment
from pyboy.utils import WindowEvent


class MarioController(MarioEnvironment):
    """
    The MarioController class represents a controller for the Mario game environment.

    You can build upon this class all you want to implement your Mario Expert agent.

    Args:
        act_freq (int): The frequency at which actions are performed. Defaults to 10.
        emulation_speed (int): The speed of the game emulation. Defaults to 0.
        headless (bool): Whether to run the game in headless mode. Defaults to False.
    """

    def __init__(
        self,
        act_freq: int = 10,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        super().__init__(
            act_freq=act_freq,
            emulation_speed=emulation_speed,
            headless=headless,
        )

        self.act_freq = act_freq

        # Example of valid actions based purely on the buttons you can press
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_button = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

    def _press_buttons(self, buttons: list[WindowEvent]) -> None:
        """Presses the given list of buttons."""
        for button in buttons:
            self.pyboy.send_input(button)
        
    def _release_buttons(self, buttons: list[WindowEvent]) -> None:
        """Releases the given list of buttons."""
        for button in buttons:
            self.pyboy.send_input(button)

    def _perform_action(self, duration: int) -> None:
        """Performs an action for a given duration."""
        for _ in range(duration):
            self.pyboy.tick()

    def run_action(self, action: int) -> None:
        """
        Executes the action based on the provided action index.

        Args:
            action (int): The index representing the action to perform.
        """
        if action == 6:
            # Running jump
            self._press_buttons([self.valid_actions[2], self.valid_actions[4]])
            self._perform_action(self.act_freq)
            self._release_buttons([self.release_button[2], self.release_button[4]])
            self._perform_action(self.act_freq)
        elif action == 7:
            # Long running jump
            self._press_buttons([self.valid_actions[2], self.valid_actions[4]])
            self._perform_action(self.act_freq * 3)
            self._release_buttons([self.release_button[2], self.release_button[4]])
            self._perform_action(self.act_freq)
        else:
            # Default action
            self._press_buttons([self.valid_actions[action]])
            self._perform_action(self.act_freq)
            self._release_buttons([self.release_button[action]])
            if action == 4:
                self.pyboy.tick()



class MarioExpert:
    """
    The MarioExpert class represents an expert agent for playing the Mario game.

    Edit this class to implement the logic for the Mario Expert agent to play the game.

    Do NOT edit the input parameters for the __init__ method.

    Args:
        results_path (str): The path to save the results and video of the gameplay.
        headless (bool, optional): Whether to run the game in headless mode. Defaults to False.
    """

    def __init__(self, results_path: str, headless=False):
        self.results_path = results_path

        self.environment = MarioController(headless=headless)

        self.video = None

    def analyse_environment(self):
        # Extracting the game area and setting up useful constants
        game_area = self.environment.game_area().tolist()
        game_area_transposed = self.environment.game_area().T.tolist()

        constants = {
            'DOWN': 0,
            'LEFT': 1,
            'RIGHT': 2,
            'UP': 3,
            'BUTTON_A': 4,
            'BUTTON_B': 5,
            'RUN_JUMP': 6,
            'LONG_JUMP': 7,
            'EMPTY': 0,
            'MARIO': 1,
            'COIN': 5,
            'MUSHROOM': 6,
            'GROUND': 10,
            'PLATFORM': 11,
            'BLOCK': 12,
            'COIN_BLOCK': 13,
            'PIPE': 14,
            'GOOMBA': 15,
            'KOOPA': 16,
            'FLY': 18,
        }

        return game_area, game_area_transposed, constants

    def find_mario(self, game_area):
        for x_index, x_row in enumerate(game_area):
            if 1 in x_row:  # MARIO
                return x_index, x_row.index(1)
        return -1, -1

    def locate_entities(self, game_area, entity_value):
        entities = []
        for x_index, x_row in enumerate(game_area):
            for y_index, cell in enumerate(x_row):
                if cell == entity_value:
                    entities.append((x_index, y_index))
        return entities

    def choose_action(self, mario_pos, game_area, game_area_transposed, constants):
        mario_x, mario_y = mario_pos
        if mario_x == -1 or mario_y == -1:
            return random.randint(0, len(self.environment.valid_actions) - 1)  # Default random action

        # Locate Goombas
        goombas = self.locate_entities(game_area, constants['GOOMBA'])
        if goombas:
            for goomba_x, goomba_y in goombas:
                # Check if Goomba is directly in Mario's path
                if mario_x + 1 == goomba_x and mario_y == goomba_y:
                    return 1  # Move Left to avoid Goomba

        # Check if Mario is stuck
        if (mario_x + 1 < len(game_area) and
            mario_y < len(game_area[0]) and
            game_area[mario_x + 1][mario_y] == constants['GROUND'] and
            (mario_y - 1 >= 0 and game_area[mario_x][mario_y - 1] == constants['GROUND']) and
            (mario_y + 1 < len(game_area[0]) and game_area[mario_x][mario_y + 1] == constants['GROUND']) and
            (mario_x - 1 >= 0 and game_area[mario_x - 1][mario_y] == constants['GROUND'])):
            
            # Move left until there's a 0 above Mario
            while mario_x - 1 >= 0 and game_area[mario_x - 1][mario_y] != constants['EMPTY']:
                mario_x -= 1
                return 1  # Move Left

            # After clearing the path, jump to the right
            return 4  # Jump

        # Check for gaps and perform a long jump if necessary
        if mario_x <= len(game_area) - 3 and mario_y <= len(game_area[0]) - 2:
            if (mario_x + 1 < len(game_area) and mario_y < len(game_area[0]) and
                mario_x + 2 < len(game_area) and mario_y < len(game_area[0]) and
                game_area[mario_x + 1][mario_y] == constants['EMPTY'] and
                game_area[mario_x + 2][mario_y] == constants['EMPTY']):
                
                # Check if landing spot is safe
                if mario_x + 3 < len(game_area) and mario_y < len(game_area[0]) and game_area[mario_x + 3][mario_y] == constants['GOOMBA']:
                    return 1  # Move Left to avoid landing on Goomba

                if mario_x + 3 < len(game_area) and mario_y < len(game_area[0]) and game_area[mario_x + 3][mario_y] in [constants['PIPE'], constants['BLOCK']]:
                    return 7  # Long Jump

                # If the gap is too wide or deep
                if mario_x + 2 < len(game_area) and mario_y < len(game_area[0]) and game_area[mario_x + 2][mario_y] == constants['EMPTY']:
                    return 6  # Long Jump into the pit and proceed

        # Adjust jumping logic to handle obstacles and gaps
        if mario_x <= len(game_area) - 3 and mario_y <= len(game_area[0]) - 2:
            if (mario_x + 1 < len(game_area) and mario_y + 2 < len(game_area[0]) and
                game_area[mario_x + 1][mario_y + 2] in [constants['PIPE'], constants['GROUND']] or
                mario_x < len(game_area) and mario_y + 2 < len(game_area[0]) and
                game_area[mario_x][mario_y + 2] in [constants['PIPE'], constants['GROUND']]):
                return 7  # Long Jump over obstacle

            if mario_x + 1 < len(game_area) and mario_y + 2 < len(game_area[0]) and (
                game_area[mario_x + 1][mario_y + 2] == constants['PIPE'] or
                game_area[mario_x + 1][mario_y + 2] == constants['GROUND']):
                return 6  # Run Jump
                
            if mario_x + 2 < len(game_area) and mario_y < len(game_area[0]) and (
                game_area[mario_x + 2][mario_y] == constants['GROUND'] and
                mario_x + 3 < len(game_area) and game_area[mario_x + 3][mario_y] == constants['GROUND']):
                return 7  # Run Jump

            if mario_x < len(game_area) and mario_y + 2 < len(game_area[0]) and (
                game_area[mario_x][mario_y + 2] == constants['GROUND'] and
                mario_x + 1 < len(game_area) and mario_y + 2 < len(game_area[0]) and
                game_area[mario_x + 1][mario_y + 2] == constants['EMPTY']):
                return 4  # Jump

        return 2  # Default move right


    def step(self):
        game_area, game_area_transposed, constants = self.analyse_environment()
        mario_pos = self.find_mario(game_area)
        action = self.choose_action(mario_pos, game_area, game_area_transposed, constants)
        self.environment.run_action(action)

    def play(self):
        """
        Do NOT edit this method.
        """
        self.environment.reset()

        frame = self.environment.grab_frame()
        height, width, _ = frame.shape

        self.start_video(f"{self.results_path}/mario_expert.mp4", width, height)

        while not self.environment.get_game_over():
            frame = self.environment.grab_frame()
            self.video.write(frame)

            self.step()

        final_stats = self.environment.game_state()
        logging.info(f"Final Stats: {final_stats}")

        with open(f"{self.results_path}/results.json", "w", encoding="utf-8") as file:
            json.dump(final_stats, file)

        self.stop_video()

    def start_video(self, video_name, width, height, fps=30):
        """
        Do NOT edit this method.
        """
        self.video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    def stop_video(self) -> None:
        """
        Do NOT edit this method.
        """
        self.video.release()
