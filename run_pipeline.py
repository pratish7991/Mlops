import subprocess

def main():
    subprocess.run(["python", "-m", "src.preprocessing.preprocess"])
    subprocess.run(["python", "-m", "src.main"])

if __name__ == "__main__":
    main()
