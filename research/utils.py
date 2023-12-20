
import os

# Resolution('1m')
def clear_terminal():
        # Check the operating system
        if os.name == "posix":  # For Linux and macOS
            os.system("clear")


def debug(*texts):
    texts = list(texts)

    display_text = "\n".join([str(text) for text in texts])
    print(display_text)
    x = input(" " )

    if x == 'x':
        clear_terminal()
        
        file_path = "logs.log"
        with open(file_path, 'w'):
            pass

        exit()
    

    