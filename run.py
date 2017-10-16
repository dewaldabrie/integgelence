from multiprocessing import Process
from tamagotchi.run import run as run_tamagotchi
from portal.run import run as run_portal


if __name__ == '__main__':
    proc_handles = []

    proc_handles.append(Process(
        target=run_tamagotchi,
    ))

    for p in proc_handles:
        p.start()

    # use the main thread to run the portal
    run_portal()