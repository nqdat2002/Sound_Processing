from waitress import serve

from app import app

def main():
    serve(app)
    print("Ended Server!!!")

if __name__ == '__main__':
    main()