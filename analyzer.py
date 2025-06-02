import cv2
import numpy as np
import librosa
from scipy.signal import detrend

def get_microsaccades(video_path):
    """Извлечение микросаккад из видео (заглушка, требует MediaPipe)."""
    # В реальной версии: использовать MediaPipe для трекинга зрачка
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Упрощённо: имитация движений зрачка
        frames.append(np.random.randn())  # Заменить на реальный трекинг
    cap.release()
    return np.array(frames)

def get_voice_features(audio_path):
    """Извлечение характеристик голоса."""
    y, sr = librosa.load(audio_path)
    # Упрощённо: спектральные характеристики
    return detrend(y[:1000])  # Ограничим для примера

def get_hrv(video_path):
    """Извлечение HRV через фотоплетизмографию (заглушка)."""
    # В реальной версии: анализ цвета кожи (rPPG)
    return np.random.randn(1000)  # Заменить на реальный алгоритм

def generate_personality_code(microsaccades, voice, hrv):
    """Создание кода личности."""
    # Упрощённый расчёт фрактальной размерности (метод Хигучи)
    def fractal_dimension(data):
        n = len(data)
        L = []
        for k in range(1, n//2):
            Lk = sum(abs(data[m + i*k] - data[m + (i-1)*k]) for m in range(k) for i in range(1, int((n - m)/k)))
            Lk *= (n - 1) / (k * int((n - m)/k))
            L.append(np.log(Lk/k) / np.log(3))  # p=3
        return np.mean(L[-10:])
    
    D_voice = fractal_dimension(voice)
 D_pupil = fractal_dimension(microsaccades)
    D_hrv = fractal_dimension(hrv)
    
    # Хешируем для уникального ID
    code = f"{int(D_voice*1e5):03d}{int(D_pupil*1e5):03d}{int(D_hrv*1e5):03d}"
    unique_id = hashlib.sha3_256(code.encode()).hexdigest()[:16]
    return unique_id, [D_voice, D_pupil, D_hrv]

def compare_personalities(id1, dims1, id2, dims2):
    """Сравнение кодов личности."""
    weights = [1.5, 2.58, 1.3]  # Фрактальные размерности
    delta_dims = [abs(dims1[i] - dims2[i])**weights[i] for i in range(3)]
    delta_id = max(delta_dims)
    
    score = 1 - delta_id / (3**-5)
    compatible = delta_id < 3**-6
    return compatible, score

# Пример использования
if __name__ == "__main__":
    # Заглушка для данных
    video1, audio1 = "user1.mp4", "user1.wav"
    video2, audio2 = "user2.mp4", "user2.wav"
    
    microsaccades1 = get_microsaccades(video1)
    voice1 = get_voice_features(audio1)
    hrv1 = get_hrv(video1)
    microsaccades2 = get_microsaccades(video2)
    voice2 = get_voice_features(audio2)
    hrv2 = get_hrv(video2)
    
    id1, dims1 = generate_personality_code(microsaccades1, voice1, hrv1)
    id2, dims2 = generate_personality_code(microsaccades2, voice2, hrv2)
    
    compatible, score = compare_personalities(id1, dims1, id2, dims2)
    print(f"Код 1: {id1}, Код 2: {id2}")
    print(f"«Я не тебя люблю, а себя в тебе»: {'Совместимы' if compatible else 'Не совместимы'} (Score: {score:.2f})")
​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
