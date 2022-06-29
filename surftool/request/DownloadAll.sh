#!/bin/bash

# from Ye Tian

# AK, Dec. 03 2020

ListSeed() {
	local _fin=$1
	local _fou=$2
	local _label=$3
	if [ -e $_fin ]; then
		more $_fin | tr " " "\n" | grep ".mseed" | grep $_label | tr "\r" " " > $_fou
		# more $_fin | tr " " "\n" | grep ".seed" | grep $_label | tr "\r" " " > $_fou
		#more $_fin | tr " " "\n" | grep ".seed" | awk -F_ '{print $1,$2}' | awk -F. '{print $1,$2,$0}' | awk '{print $2,$3,$4"_"$5}' | sort -M -k2 | sort -s -g -k1 | awk '{print $3}' | tr "\r" " " > $_fou
	fi
}

CheckFList() {
	local fin=$1
	# return 1 if input file doesnot exist !
	if [ ! -e $fin ]; then
		echo 1
		return
	fi
	# check if all downloaded
	local vret=1
	while read file; do
		#echo $file 1>&2
		if [ ! -e $file ]; then
			vret=0
			break
		fi
	done < $fin
	echo $vret
}


DownloadFromFile() {
	local _flst=$1
	local _address=$2
	while read file; do
		local _furl=`echo ${_address}"/"${file}`
		wget -c ${_furl}
	done < $_flst
}


### main ###
address=ftp://ftp.iris.washington.edu/pub/userdata/LiliFeng

# address=ftp://ftp.iris.washington.edu/pub/userdata/ChuanmingLiu
# address=ftp://ftp.iris.washington.edu/pub/userdata/Hongda/
# address=ftp://ftp.iris.washington.edu/pub/userdata/Ye_Tian/
maxwait=1000

NfileAll=`grep '\.mseed' .listing  | wc -l`
while [ true ]; do
	# grap file list
	flst=.listing
	wget -A seed -nd -c -r --no-remove-listing -p -E -k -K -np $address/$flst 
	# produce seed list (sorted by year.month)
	slst=.fseeds
	ListSeed $flst $slst CML
	# check 
	if [ -e $flst ] && [ `CheckFList $slst` == 1 ]; then 
		let nwait++
		if [ $nwait -gt $maxwait ]; then break; fi
		echo "All seeds downloaded. Sleep for 100 sec before starting the next("$nwait"/"$maxwait") cycle..."
		sleep 100
	else
		#wget -A seed -nd -c -r --no-remove-listing -p -E -k -K -np $address
		DownloadFromFile $slst $address
		echo "Current cycle completed!"
		nwait=0
		sleep 3
	fi
done
