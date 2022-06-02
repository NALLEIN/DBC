# for file in `find /media/jianghao/Samsung_T5/bilibili_test | grep ".*origvideo"`
for file in `cat test_videos.txt`
do
    echo $(basename $file .origvideo)
    tmppath="/home/jianghao/Code/Graduation/4k1/videos/"
    echo $tmppath$(basename $file .origvideo)
    mkdir $tmppath$(basename $file .origvideo)

    /home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -i $file -pix_fmt yuv420p -frames 1000 -vf crop=3840:2144:0:0 decode.yuv -y # > /dev/null 2>&1

	/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 3840x2144 -pix_fmt yuv420p -r 25 -i decode.yuv -pix_fmt yuv420p -s 1920x1072 \
	-vf scale=1920x1072 -sws_flags lanczos lanczos_down.yuv -y

    /home/jianghao/anaconda3/envs/torch/bin/python /home/jianghao/Code/Graduation/4k1/codes/test_forward_yuv.py \
    -opt /home/jianghao/Code/Graduation/4k1/codes/options/test/test_EDSR_forward.yml
	for qp in {22..37..5}
	do
        /home/jianghao/Code/Graduation/VVNC/vvenc-1.4.0/bin/release-static/vvencFFapp -s 1920x1072 -fr 25 -i decode_net.yuv \
        --preset faster -q ${qp} -b transcode_net.266 > $tmppath$(basename $file .origvideo)/transcode_net_${qp} 2>&1

        /home/jianghao/Code/Graduation/VVNC/VTM/bin/DecoderAppStatic -b transcode_net.266 -o transcode_net.yuv -d 8 > $tmppath$(basename $file .origvideo)/decode_net_${qp} 2>&1

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1072 -pix_fmt yuv420p -r 25 -i transcode_net.yuv -pix_fmt yuv420p -s 3840x2144 \
		-vf scale=3840x2144 -sws_flags bicubic net_cubic.yuv -y

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -r 25 -i decode.yuv -s 3840x2144 -pix_fmt yuv420p -r 25 -i net_cubic.yuv \
		-filter_complex "[1:v]scale=3840x2144[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .origvideo)/psnr_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_psnr_net_${qp} 2>&1
		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -r 25 -i decode.yuv -s 3840x2144 -pix_fmt yuv420p -r 25 -i net_cubic.yuv \
		-filter_complex "[1:v]scale=3840x2144[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .origvideo)/ssim_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_ssim_net_${qp} 2>&1

        /home/jianghao/Code/Graduation/VVNC/vvenc-1.4.0/bin/release-static/vvencFFapp -s 1920x1072 -fr 25 -i lanczos_down.yuv \
        --preset faster -q ${qp} -b transcode_lanczos.266 > $tmppath$(basename $file .origvideo)/transcode_lanczos_${qp} 2>&1

        /home/jianghao/Code/Graduation/VVNC/VTM/bin/DecoderAppStatic -b transcode_lanczos.266 -o transcode_lanczos.yuv -d 8 > $tmppath$(basename $file .origvideo)/decode_lanczos_${qp} 2>&1

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg -s 1920x1072 -pix_fmt yuv420p -r 25 -i transcode_lanczos.yuv -pix_fmt yuv420p -s 3840x2144 \
		-vf scale=3840x2144 -sws_flags bicubic lanczos_cubic.yuv -y

		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -r 25 -i decode.yuv -s 3840x2144 -pix_fmt yuv420p -r 25 -i lanczos_cubic.yuv \
		-filter_complex "[1:v]scale=3840x2144[scale];[scale][0:v]psnr=stats_file="$tmppath$(basename $file .origvideo)/psnr_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_psnr_lanczos_${qp} 2>&1
		/home/jianghao/Documents/ffmpeg/build/bin/ffmpeg  -s 3840x2144 -pix_fmt yuv420p -r 25 -i decode.yuv -s 3840x2144 -pix_fmt yuv420p -r 25 -i lanczos_cubic.yuv \
		-filter_complex "[1:v]scale=3840x2144[scale];[scale][0:v]ssim=stats_file="$tmppath$(basename $file .origvideo)/ssim_net_${qp} \
		-f null /dev/null > $tmppath$(basename $file .origvideo)/Avg_ssim_lanczos_${qp} 2>&1
	done
	python BD-rate.py -path $tmppath$(basename $file .origvideo) >> BD-Rate-260000-266-22-37.log
done
