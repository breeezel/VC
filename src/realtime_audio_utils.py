import sounddevice as sd
import numpy as np # Не используется здесь напрямую, но часто нужен вместе с sounddevice

def list_audio_devices():
    """Выводит список доступных аудиоустройств ввода и вывода."""
    print("Доступные аудиоустройства:")
    devices = sd.query_devices()
    if not devices:
        print("Аудиоустройства не найдены.")
        return []
    for i, device in enumerate(devices):
        print(f"  ID {i}: {device['name']}")
        print(f"    Макс. каналов ввода: {device['max_input_channels']}")
        print(f"    Макс. каналов вывода: {device['max_output_channels']}")
        print(f"    Стандартная частота дискр.: {device['default_samplerate']}")
    return devices

def select_device_id(prompt_message="Выберите ID устройства: ", kind="ввода/вывода"):
    """
    Запрашивает у пользователя ID аудиоустройства из списка.
    Args:
        prompt_message (str): Сообщение для пользователя.
        kind (str): Тип устройства (для контекста в сообщении).
    Returns:
        int: Выбранный ID устройства, или None при неверном вводе.
    """
    while True:
        try:
            device_id_str = input(prompt_message)
            if not device_id_str:
                print(f"Для {kind} не выбрано устройство. Укажите корректный ID или проверьте конфигурацию.")
                return None
            device_id = int(device_id_str)
            # Здесь можно добавить проверку на существование устройства с таким ID,
            # сравнив с len(sd.query_devices()), но для простоты опускаем.
            if device_id >= 0:
                return device_id
            else:
                print("Неверный ID устройства. Введите неотрицательное целое число.")
        except ValueError:
            print("Неверный ввод. Введите числовой ID.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None
