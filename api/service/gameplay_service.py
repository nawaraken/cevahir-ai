import random
import logging
import numpy as np
import requests
from config.parameters import MODEL_ROUTE_BASE_URL, MAX_GAME_STEPS, INPUT_DIM, BOARD_SIZE

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO)
game_logger = logging.getLogger("gameplay_service")
file_handler = logging.FileHandler('game.log', mode='w')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
game_logger.addHandler(file_handler)


class GameplayService:
    def __init__(self, max_steps=MAX_GAME_STEPS):
        """
        GameplayService sınıfı, modelin oyun oynama yeteneklerini test etmek ve geliştirmek için tasarlandı.

        Args:
            max_steps (int): Oyunun maksimum adım sayısı.
        """
        self.max_steps = max_steps
        self.board_size = BOARD_SIZE
        self.game_state = None
        self.replay_buffer = []
        self.check_model_loaded()

    def check_model_loaded(self):
        """
        Modelin `model_routes` üzerinden yüklü olup olmadığını kontrol eder.
        """
        try:
            response = requests.post(f"{MODEL_ROUTE_BASE_URL}/load")
            if response.status_code == 200:
                game_logger.info("Model başarıyla yüklendi.")
            else:
                raise RuntimeError("Model yüklenemedi. Lütfen model_routes kontrol edin.")
        except Exception as e:
            game_logger.error(f"Model yüklenirken hata: {str(e)}")
            raise RuntimeError("GameplayService başlatılamadı. Model yüklenemedi.")

    def initialize_game(self):
        """
        Oyunun başlangıç durumunu ayarlar.
        """
        self.game_state = np.zeros((self.board_size, self.board_size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        game_logger.info("Oyun başarıyla başlatıldı.")

    def add_random_tile(self):
        """
        Boş bir hücreye rastgele bir 2 veya 4 değeri ekler.
        """
        empty_cells = list(zip(*np.where(self.game_state == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.game_state[x, y] = 2 if random.random() < 0.9 else 4
            game_logger.debug(f"Yeni taş eklendi: ({x}, {y}) -> {self.game_state[x, y]}")

    def move(self, direction):
        """
        Oyunun durumunu verilen yönde günceller.
        Args:
            direction (int): 0 = yukarı, 1 = sağ, 2 = aşağı, 3 = sol
        """
        original_state = self.game_state.copy()
        if direction == 0:
            self.game_state = self._move_up()
        elif direction == 1:
            self.game_state = self._move_right()
        elif direction == 2:
            self.game_state = self._move_down()
        elif direction == 3:
            self.game_state = self._move_left()

        if not np.array_equal(original_state, self.game_state):
            self.add_random_tile()

    def _move_up(self):
        return self._transpose_and_merge(self.game_state)

    def _move_down(self):
        return np.flipud(self._transpose_and_merge(np.flipud(self.game_state)))

    def _move_left(self):
        return self._merge_rows(self.game_state)

    def _move_right(self):
        return np.fliplr(self._merge_rows(np.fliplr(self.game_state)))

    def _merge_rows(self, state):
        new_state = np.zeros_like(state)
        for i in range(self.board_size):
            non_zero = state[i][state[i] != 0]
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j < len(non_zero) - 1 and non_zero[j] == non_zero[j + 1]:
                    merged.append(non_zero[j] * 2)
                    skip = True
                else:
                    merged.append(non_zero[j])
            new_state[i, :len(merged)] = merged
        return new_state

    def _transpose_and_merge(self, state):
        return self._merge_rows(state.T).T

    def play(self):
        """
        Modelin oyunu oynamasını sağlar.
        """
        self.initialize_game()
        total_reward = 0
        successful_moves = 0
        game_logger.info("Oyun başlıyor.")

        for step in range(self.max_steps):
            current_state = self.expand_input(self.game_state)
            action = self.get_action_from_model(current_state)

            previous_state = self.game_state.copy()
            self.move(action)
            reward = self.calculate_reward(previous_state)

            total_reward += reward
            if reward > 0:
                successful_moves += 1

            self.update_replay_buffer(current_state, action, reward)

            game_logger.info(f"Adım {step + 1}: Model {['Yukarı', 'Sağ', 'Aşağı', 'Sol'][action]} yönünde hareket etti.")
            game_logger.info(f"Oyun durumu:\n{self.game_state}")
            game_logger.info(f"Adım ödülü: {reward} | Toplam ödül: {total_reward} | Başarılı hamleler: {successful_moves}")

            if not self.can_move():
                game_logger.info(f"Oyun sona erdi! Toplam ödül: {total_reward} | Başarılı hamleler: {successful_moves}")
                break

    def get_action_from_model(self, state):
        """
        Modelden aksiyon alır.
        """
        try:
            response = requests.post(
                f"{MODEL_ROUTE_BASE_URL}/forward",
                json={"inputs": state.tolist()}
            )
            if response.status_code == 200:
                outputs = response.json()["outputs"]
                return int(np.argmax(outputs["main_output"]))
            else:
                raise RuntimeError(f"Modelden aksiyon alınamadı: {response.text}")
        except Exception as e:
            game_logger.error(f"Modelden aksiyon alınırken hata: {str(e)}")
            raise RuntimeError("Modelden aksiyon alınamadı.")

    def expand_input(self, game_state):
        """
        Giriş durumunu model için uygun boyuta genişletir.
        """
        flat_state = game_state.flatten()
        padded_state = np.pad(flat_state, (0, INPUT_DIM - len(flat_state)), mode='constant')
        return np.expand_dims(padded_state, axis=0)

    def calculate_reward(self, previous_state):
        """
        Ödül hesaplama.
        """
        if not np.array_equal(previous_state, self.game_state):
            return 1
        return -1

    def update_replay_buffer(self, state, action, reward):
        """
        Replay buffer'ı günceller.
        """
        self.replay_buffer.append((state, action, reward))
        if len(self.replay_buffer) > 1000:
            self.replay_buffer.pop(0)

    def can_move(self):
        """
        Oyunun devam edip edemeyeceğini kontrol eder.
        """
        for direction in range(4):
            if not np.array_equal(self.game_state, self.move_and_check(direction)):
                return True
        return False

    def move_and_check(self, direction):
        """
        Belirtilen yönde hareketi simüle eder.
        """
        if direction == 0:
            return self._move_up()
        elif direction == 1:
            return self._move_right()
        elif direction == 2:
            return self._move_down()
        elif direction == 3:
            return self._move_left()
        return self.game_state.copy()
