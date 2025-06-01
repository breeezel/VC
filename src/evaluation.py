import numpy as np
import librosa
import logging
from dtw import dtw # Используем пакет dtw-python

# Настройка логгера для этого модуля
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Установим INFO, чтобы видеть сообщения о расчете
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def calculate_mcd(converted_audio_data, target_audio_data, sample_rate,
                  mfcc_params=None, dtw_enabled=True):
    """
    Вычисляет Mel Cepstral Distortion (MCD) между двумя аудиосигналами.
    Args:
        converted_audio_data (np.ndarray): Синтезированные аудиоданные.
        target_audio_data (np.ndarray): Истинные (целевые) аудиоданные.
        sample_rate (int): Частота дискретизации аудио.
        mfcc_params (dict, optional): Параметры для извлечения MFCC.
            Пример: {'n_mfcc': 24, 'n_fft': 1024, 'hop_length': 256}
        dtw_enabled (bool): Использовать ли Dynamic Time Warping для выравнивания.
    Returns:
        float: Значение MCD. Возвращает float('inf') при ошибке.
    """
    logger.info(f"Начало вычисления MCD. DTW включен: {dtw_enabled}")

    default_mfcc_params = {'n_mfcc': 24, 'n_fft': 1024, 'hop_length': 256, 'win_length': None}
    if mfcc_params:
        default_mfcc_params.update(mfcc_params)

    try:
        # Извлечение MFCC (исключая C0, т.е. [1:])
        mfcc_converted = librosa.feature.mfcc(y=converted_audio_data, sr=sample_rate, **default_mfcc_params)[1:]
        mfcc_target = librosa.feature.mfcc(y=target_audio_data, sr=sample_rate, **default_mfcc_params)[1:]

        logger.debug(f"Форма MFCC (конверт.): {mfcc_converted.shape}, Форма MFCC (цель): {mfcc_target.shape}")

        if mfcc_converted.shape[1] == 0 or mfcc_target.shape[1] == 0:
            logger.warning("Один из MFCC массивов пуст. Невозможно вычислить MCD.")
            return float('inf')

        if dtw_enabled:
            # Выравнивание с помощью DTW
            # dtw-python ожидает (N, D) где N - количество кадров, D - размерность признаков
            # librosa.feature.mfcc возвращает (D, N), поэтому транспонируем (.T)
            try:
                # Используем евклидово расстояние для стоимостной функции DTW
                # dtw_result = dtw(mfcc_converted.T, mfcc_target.T, dist_method='euclidean', keep_internals=True)
                # path_x, path_y = dtw_result.index1, dtw_result.index2
                # aligned_mfcc_converted = mfcc_converted[:, path_x]
                # aligned_mfcc_target = mfcc_target[:, path_y]

                # Альтернативный подход с dtw-python: вычисление расстояния напрямую
                # Для MCD обычно суммируют евклидовы расстояния вдоль пути DTW
                # dtw() возвращает объект с атрибутом .distance (сумма стоимостей пути)
                # или .normalizedDistance. Для MCD нам нужен доступ к выровненным векторам или ручной подсчет.

                # Простой способ с dtw-python: выровнять и затем считать вручную
                # Для MCD классически используется евклидово расстояние между выровненными векторами.
                # dtw-python может быть сложен для получения именно выровненных последовательностей в явном виде.
                # Вместо этого, можно использовать реализацию DTW, которая возвращает путь,
                # или, если dtw.distance это сумма евклидовых расстояний по пути, это можно адаптировать.
                # Для простоты, если dtw-python не дает выровненные последовательности легко,
                # мы можем использовать более простую форму DTW или другую библиотеку.
                # Однако, dtw-python.dtw(x,y).path дает индексы.

                alignment = dtw(mfcc_converted.T, mfcc_target.T, keep_internals=False,
                                step_pattern="symmetric2", open_end=False, open_begin=False) # Используем стандартные параметры

                path_x = alignment.index1
                path_y = alignment.index2

                aligned_mfcc_converted = mfcc_converted[:, path_x]
                aligned_mfcc_target = mfcc_target[:, path_y]

                logger.debug(f"Форма MFCC после DTW (конверт.): {aligned_mfcc_converted.shape}, (цель): {aligned_mfcc_target.shape}")

            except Exception as e_dtw:
                logger.error(f"Ошибка DTW: {e_dtw}. Возвращаем MCD без DTW.", exc_info=True)
                # Обрезаем до минимальной длины при ошибке DTW
                min_len = min(mfcc_converted.shape[1], mfcc_target.shape[1])
                aligned_mfcc_converted = mfcc_converted[:, :min_len]
                aligned_mfcc_target = mfcc_target[:, :min_len]
        else:
            # Обрезаем до минимальной длины, если DTW отключен
            min_len = min(mfcc_converted.shape[1], mfcc_target.shape[1])
            aligned_mfcc_converted = mfcc_converted[:, :min_len]
            aligned_mfcc_target = mfcc_target[:, :min_len]
            logger.debug(f"Форма MFCC без DTW (конверт.): {aligned_mfcc_converted.shape}, (цель): {aligned_mfcc_target.shape}")

        if aligned_mfcc_converted.shape[1] == 0:
            logger.warning("После выравнивания/обрезки длина MFCC стала 0. Невозможно вычислить MCD.")
            return float('inf')

        # Вычисление MCD
        # Сумма евклидовых расстояний по каждому кадру, усредненная по количеству кадров
        diff = aligned_mfcc_target - aligned_mfcc_converted
        dist = np.sqrt(np.sum(diff**2, axis=0)) # Евклидово расстояние для каждого кадра
        mcd_val = np.mean(dist)

        # Традиционный множитель для MCD (хотя иногда используется и без него)
        mcd_val = (10.0 / np.log(10.0)) * mcd_val # ~4.34
        # Иногда встречается еще sqrt(2) в формуле, но это зависит от определения.
        # mcd_val = (10.0 / np.log(10.0)) * np.sqrt(2) * mcd_val # ~6.14 * mcd_val_unscaled

        logger.info(f"MCD успешно вычислен: {mcd_val:.4f}")
        return mcd_val

    except Exception as e:
        logger.error(f"Ошибка при вычислении MCD: {e}", exc_info=True)
        return float('inf') # Возвращаем бесконечность при ошибке


def calculate_f0_rmse(converted_audio_data, target_audio_data, sample_rate, f0_params=None):
    """
    Вычисляет F0 Root Mean Squared Error (RMSE) между двумя аудиосигналами.
    Args:
        converted_audio_data (np.ndarray): Синтезированные аудиоданные.
        target_audio_data (np.ndarray): Истинные (целевые) аудиоданные.
        sample_rate (int): Частота дискретизации аудио.
        f0_params (dict, optional): Параметры для извлечения F0.
            Пример: {'fmin_hz': 60, 'fmax_hz': 750, 'hop_length': 256}
    Returns:
        float: Значение F0-RMSE в Гц. Возвращает float('inf') при ошибке.
    """
    logger.info("Начало вычисления F0-RMSE.")

    default_f0_params = {'fmin': librosa.note_to_hz('C2'), 'fmax': librosa.note_to_hz('C7'),
                         'hop_length': 256} # hop_length должен соответствовать MFCC для легкого сравнения
    if f0_params:
        # Переводим fmin_hz, fmax_hz в fmin, fmax если они есть
        if 'fmin_hz' in f0_params: f0_params['fmin'] = f0_params.pop('fmin_hz')
        if 'fmax_hz' in f0_params: f0_params['fmax'] = f0_params.pop('fmax_hz')
        default_f0_params.update(f0_params)

    try:
        # Извлечение F0
        f0_converted, _, _ = librosa.pyin(converted_audio_data, **default_f0_params)
        f0_target, _, _ = librosa.pyin(target_audio_data, **default_f0_params)

        logger.debug(f"Длина F0 (конверт.): {len(f0_converted)}, (цель): {len(f0_target)}")

        # Обеспечиваем одинаковую длину для f0_converted и f0_target
        if f0_converted is not None and f0_target is not None: # Дополнительная проверка, что они не None
            min_len = min(len(f0_converted), len(f0_target))
            f0_converted = f0_converted[:min_len]
            f0_target = f0_target[:min_len]
        else:
            # Если один из них None, F0-RMSE вычислить невозможно или будет некорректным
            logger.warning("Не удалось извлечь F0 для одного или обоих аудиофайлов. F0-RMSE будет inf.")
            return float('inf')

        # Обработка NaN (неозвученные кадры) - оставляем только те кадры, где оба F0 определены
        voiced_mask = ~np.isnan(f0_converted) & ~np.isnan(f0_target)

        f0_converted_voiced = f0_converted[voiced_mask]
        f0_target_voiced = f0_target[voiced_mask]

        if len(f0_converted_voiced) == 0 or len(f0_target_voiced) == 0:
            logger.warning("Нет общих озвученных кадров для F0. Невозможно вычислить F0-RMSE.")
            return float('inf') # Или 0, или специальное значение

        logger.debug(f"Количество общих озвученных кадров для F0: {len(f0_converted_voiced)}")

        # Вычисление RMSE
        f0_rmse_val = np.sqrt(np.mean((f0_converted_voiced - f0_target_voiced)**2))

        logger.info(f"F0-RMSE успешно вычислен: {f0_rmse_val:.2f} Гц")
        return f0_rmse_val

    except Exception as e:
        logger.error(f"Ошибка при вычислении F0-RMSE: {e}", exc_info=True)
        return float('inf')
