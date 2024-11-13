# demuxnet/__main__.py

from utils import read_rds


def main():
    print("DemuxNet is running!")

    # Your main script logic here
    data=read_rds("/home/wuyou/Projects/scRNA-seq/20230506_full_matrix.rds")
    #print(data)

    

if __name__ == "__main__":
    main()


