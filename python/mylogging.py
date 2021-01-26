def log_error(s):
    with open("error_log.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write(s)
        file_object.write("\n")
