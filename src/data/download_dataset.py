import sys
import subprocess


def download_audio(links, out_label):
    for link in links:
        # Setting up output location
        output_location = "-o \"../../data/raw/" + out_label + "/%(title)s.%(ext)s\" "

        # youtube-dl configuration file location
        config_args = "--config-location ../../external/youtube-dl.conf"

        # subprocess call to run youtube-dl
        subprocess.run("../../external/youtube-dl " + config_args + " " + output_location + " \"" + link + "\"")


def usage():
    print("Usage: python3 download_dataset.py input_file1 input_file2 ... input_fileN\n\n"
          "input_file is a list of youtube playlist/video links. \n Can use # for comments")


def read_links(filename):
    links = list()
    with open(filename) as f:
        for line in f:
            # Ignoring comments
            if line[0] != '#':
                links.append(line.replace("\n", ""))
    return links


def main():
    if len(sys.argv) < 2:
        usage()
    else:
        # Reading file contents and sending for download
        for filename in sys.argv[1:]:
            links = read_links(filename)

            # Extracting file name for output directory location
            out = filename.split("/")[-1]
            download_audio(links, out)


if __name__ == '__main__':
    main()
