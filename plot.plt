set terminal pngcairo enhanced size 2200,600

set xrange[0:2000]
set yrange[0:400]

set cbrange[0.0000000001:1]

set logscale cb

! mkdir /tmp/gnuplot-tmp

skip=100
do for [i=0:29900:skip] {
	set output sprintf("/tmp/gnuplot-tmp/tmp%04d.png",i/skip+1)
	plot 'data/'.i.'.csv' matrix with image
}

! ffmpeg -i /tmp/gnuplot-tmp/tmp%04d.png -y -c:v mpeg4 -q:v 1 test.avi
# ! rm -r /tmp/gnuplot-tmp
