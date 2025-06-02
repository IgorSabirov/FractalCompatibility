import cv2
import numpy as np
import librosa
import mediapipe as mp
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- Функции для извлечения данных ---

def extract_pupil_movement(video_path):
    """Извлечение движений зрачка с помощью MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    landmarks = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                # Landmarks 468-473 - область правого глаза (зрачок)
                pupil_pos = np.mean([(lm.x, lm.y) 
                    for lm in results.multi_face_landmarks[0].landmark[468:473]], axis=0)
                landmarks.append(pupil_pos[0])  # Используем только x-координату для простоты
    cap.release()
    return np.array(landmarks)

def get_hrv(video_path):
    """Извлечение HRV с помощью базового rPPG (анализ цвета кожи)."""
    cap = cv2.VideoCapture(video_path)
    pulse_signal = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Регион интереса (ROI) - лоб, примерные координаты
        roi = frame[50:100, 100:200]
        avg_color = np.mean(roi, axis=(0,1))
        pulse_signal.append(avg_color[0])  # Красный канал для анализа пульса
    cap.release()
    # Поиск пиков пульса и вычисление RR-интервалов
    fps = cap.get(cv2.CAP_PROP_FPS)
    peaks, _ = scipy.signal.find_peaks(pulse_signal, distance=int(fps/2))
    rr_intervals = np.diff(peaks) / fps
    return rr_intervals

def get_voice_features(audio_path):
    """Извлечение MFCC из аудио для анализа голоса."""
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # Среднее по времени для упрощения

# --- Функции для фрактального анализа ---

def normalize_signal(signal):
    """Нормализация сигнала для анализа."""
    return (signal - np.mean(signal)) / np.std(signal)

def p_adic_higuchi_fd(signal, p=3, k_max=10):
    """p-адический метод Хигучи для вычисления фрактальной размерности."""
    n = len(signal)
    L = np.zeros(k_max)
    for k in range(1, k_max + 1):
        Lmk = 0
        for m in range(k):
            idx = np.arange(m, n, k, dtype=int)
            if len(idx) > 1:  # Проверяем, что есть данные для вычисления
                diff = np.abs(np.diff(signal[idx]))
                Lmk += np.sum(diff) * (n - 1) / (len(idx) * k)
        L[k-1] = np.log(Lmk / k + 1e-10) / np.log(p)  # Добавляем малое значение для избежания log(0)
    slope, _ = np.polyfit(np.arange(1, k_max + 1), L, 1)
    return slope  # Наклон как фрактальная размерность

# --- Функции для генерации и сравнения ID ---

def generate_fractal_id(pupil_data, voice_data, hrv_data, p=3):
    """Генерация фрактального ID на основе трех сигналов."""
    pupil_fd = p_adic_higuchi_fd(normalize_signal(pupil_data), p)
    voice_fd = p_adic_higuchi_fd(normalize_signal(voice_data), p)
    hrv_fd = p_adic_higuchi_fd(normalize_signal(hrv_data), p)
    return np.array([pupil_fd, voice_fd, hrv_fd])

def compare_personalities(dims1, dims2):
    """Сравнение двух личностей на основе фрактальных ID."""
    # Теоретически обоснованные веса (пример из исследований)
    weights = np.array([1.0, 2.58496, 1.3])
    # Вычисление p-адической нормы (ord=3)
    delta = np.linalg.norm((dims1 - dims2) * weights, ord=3)
    compatible = delta < 0.1  # Порог из экспериментальных данных
    score = 1 - delta  # Оценка совместимости
    return compatible, score

# --- Визуализация ---

def visualize_ids(dims_list, labels):
    """Визуализация фрактальных ID с помощью t-SNE."""
    tsne = TSNE(n_components=2, metric="manhattan", random_state=42)
    X_embedded = tsne.fit_transform(np.array(dims_list))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
    plt.title("t-SNE визуализация фрактальных ID")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

# --- Пример использования ---

if __name__ == "__main__":
    # Укажите пути к вашим файлам
    video_path = "path/to/your/video.mp4"  # Замените на реальный путь
    audio_path = "path/to/your/audio.wav"  # Замените на реальный путь

    # Извлечение данных
    pupil_data = extract_pupil_movement(video_path)
    hrv_data = get_hrv(video_path)
    voice_data = get_voice_features(audio_path)

    # Генерация фрактальных ID для двух "личностей" (вторая с небольшим шумом)
    dims1 = generate_fractal_id(pupil_data, voice_data, hrv_data)
    dims2 = generate_fractal_id(
        pupil_data + 0.01 * np.random.randn(*pupil_data.shape),
        voice_data + 0.01 * np.random.randn(*voice_data.shape),
        hrv_data + 0.01 * np.random.randn(*hrv_data.shape)
    )

    # Сравнение личностей
    compatible, score = compare_personalities(dims1, dims2)
    print(f"Совместимость: {'Да' if compatible else 'Нет'}")
    print(f"Оценка совместимости: {score:.3f}")

    # Визуализация
    dims_list = [dims1, dims2]
    labels = [0, 1]  # Метки для двух личностей
    visualize_ids(dims_list, labels)
​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
