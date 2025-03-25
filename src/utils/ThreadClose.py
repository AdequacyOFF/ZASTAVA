def end_thread(thread):
    if thread and thread.isRunning():
        thread.quit()  # Остановка event loop потока
        # thread.wait()  # Ожидание завершения