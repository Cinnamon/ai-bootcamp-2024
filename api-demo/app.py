from multiprocessing import Process

from fastapi_backend import main as run_backend
from gradio_frontend import main as run_frontend


if __name__ == "__main__":
    backend_process = Process(target=run_backend)
    backend_process.start()

    frontend_process = Process(target=run_frontend)
    frontend_process.start()

    backend_process.join()
    frontend_process.join()
