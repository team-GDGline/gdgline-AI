import os
import signal
import subprocess

def stop_torchserve():
    # TorchServe 종료 시도
    try:
        subprocess.run(['torchserve', '--stop'], check=True)
        print("TorchServe가 종료되었습니다.")
    except subprocess.CalledProcessError:
        print("TorchServe 종료 중 오류가 발생했습니다.")

    # .model_server.pid 파일 경로 설정
    pid_file = os.path.join(os.path.expanduser('~'), '.model_server.pid')

    # PID 파일이 존재하는지 확인
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        try:
            # 프로세스 종료 시도
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=True)
            else:  # Unix 계열
                os.kill(pid, signal.SIGTERM)
            print(f"PID {pid}의 프로세스를 종료했습니다.")
        except Exception as e:
            print(f"PID {pid}의 프로세스 종료 중 오류가 발생했습니다: {e}")
        finally:
            # PID 파일 삭제
            os.remove(pid_file)
            print(f"{pid_file} 파일이 삭제되었습니다.")
    else:
        print(f"{pid_file} 파일이 존재하지 않습니다.")

    # 특정 포트를 사용하는 프로세스 종료 (예: 8080 포트)
    port = 8080
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if f":{port} " in line:
                    pid = int(line.strip().split()[-1])
                    subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=True)
                    print(f"포트 {port}를 사용하는 PID {pid}의 프로세스를 종료했습니다.")
        else:  # Unix 계열
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            for line in result.stdout.splitlines()[1:]:
                parts = line.split()
                pid = int(parts[1])
                os.kill(pid, signal.SIGTERM)
                print(f"포트 {port}를 사용하는 PID {pid}의 프로세스를 종료했습니다.")
       
    except Exception as e:
        print(f"포트 {port}를 사용하는 프로세스 종료 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    stop_torchserve()
