import os

if __name__ == "__main__":
    # file = open("data/coordinates_10.txt", 'a')
    #
    # for i in range(11, 35):
    #     with open(f"./data/coordinates_{i}.txt", "r") as f:
    #         file.write(f.read())
    #
    # file.close()

    tot_pts = 0
    nb_files = 0
    for filename in os.listdir("data/"):
        with open(f"data/{filename}", "r") as f:
            tot_pts += len(f.readlines())
            nb_files += 1

    print(tot_pts / nb_files)
