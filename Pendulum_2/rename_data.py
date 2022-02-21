import os

if __name__ == "__main__":
    # file = open("Data/coordinates_10.txt", 'a')
    #
    # for i in range(11, 35):
    #     with open(f"./Data/coordinates_{i}.txt", "r") as f:
    #         file.write(f.read())
    #
    # file.close()

    tot_pts = 0
    nb_files = 0
    for filename in os.listdir("./Data/"):
        with open(f"./Data/{filename}", "r") as f:
            tot_pts += len(f.readlines())
            nb_files += 1

    print(tot_pts / nb_files)
