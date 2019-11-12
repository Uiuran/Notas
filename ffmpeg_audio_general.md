# ffmpeg

Sintax:

ffmpeg [input options] -i input_file_or_stream [output options] output_file_stream

Ex:

-ac: number of channels

-ar: sampling_rate

-acodec: audio codec, use ffmpeg -codecs to list all available

Do ffmpeg -formats to see which formats are allowed to write the encoded packets of data. Use Audacity to verify transformation integrity.


# Full detailed transcoding process

![transcoding](/transcoding.png)

# References

- Wav: https://en.wikipedia.org/wiki/WAV#Specification

- ffmpeg: https://ffmpeg.org/ffmpeg.html#Audio-Options

- Digital Sound Source: http://digitalsoundandmusic.com/curriculum/

- Scaling filter (thanks glerm): https://trac.ffmpeg.org/wiki/Scaling
