from multiprocessing import Process
from tamagotchi.run import run as run_tamagotchi


if __name__ == '__main__':
    proc_handles = []

    proc_handles.append(Process(
        target=run_tamagotchi,
    ))

    for p in proc_handles:
        p.start()