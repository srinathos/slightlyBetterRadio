from pytube import YouTube
from pytube import Playlist
from moviepy.editor import *
import sys


def playlist_download():
    ad_playlist = Playlist("https://www.youtube.com/watch?v=1hByG29fne0&list=PLdb2VaO4d-cRUkibgB2M06tLKN4Lom98S")


def audio_download(url):
    yt = YouTube(url)
    stream = yt.streams.first()
    stream.download()
    audio = yt.streams.filter(only_audio=True).all()
    audio[0].download()


def usage():
    print("Usage: python3 radio.py input_file1 input_file2 ... input_fileN")


def read_links(filename):
    links = list()
    with open(filename) as f:
        for line in f:
            if line[0] != '#':
                links.append(line)
    return links


def main():
    if len(sys.argv) < 2:
        usage()
    else:
        # Reading file contents and sending for download
        for filename in sys.argv[1:]:
            links = read_links(filename)
        url = "https://www.youtube.com/watch?v=QnLMW08EdtE"
        audio_download(url)


if __name__ == '__main__':
    main()

