def Rabatel16_folder():
    hostname = socket.gethostname()
    # We don't want to publish these files for the moment :)
    if hostname == "ige-osugb1-p48":
        return "/home/gossarda/Téléchargements/geodetic_Rabatel16/"
    elif "bigfoot" in hostname:
        return "/home/gossarda/geodetic_Rabatel16/"
    elif hostname == "63bceb1ea564":
        return "/workspace/geodetic_Rabatel16/"
    elif hostname == "ige-calcul1" or hostname == "ige-calcul3":
        return "/home/gossarda/geodetic_Rabatel16/"
    else:
        raise ValueError(f"Unknown host {hostname}")
