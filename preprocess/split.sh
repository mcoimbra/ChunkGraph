echo 'Input: [preprocess_path]/split.sh [input_txt] [output_dir] [num_split_files] [name]'

if [ ! $# -eq 4 ];
then
	echo Input wrong
	exit
fi

echo $@

file=$1
outputPath=$2
file_count=$3
name=$4

if [ ! -f $file ];
then
	echo File **$file** does not exist
	exit
fi


edge_count=$(wc -l $file | awk -F " " '{print $1}')
split_line=$((edge_count/file_count))


###################
echo Splitting the large file into ~200 smaller files 

if [ ! $((split_line * file_count)) -eq $edge_count ]
then
    file_count=$((file_count+1))
fi
echo "File $file, split to $file_count files, each contains $split_line" 

prefix=${outputPath}/${name}
split -l $split_line "$file" $file


#####################
echo Renaming the split file
echo $PWD

m=0
for r1 in {a..z};
do
	for r2 in {a..z};
	do
		name="$file""$r1""$r2"

		if [ ! -f $name ];then
			continue
		fi
		new="$prefix"-$(printf "%05d" $m)
		echo mv $name $new
		mv $name $new
		m=$((m+1))
	done
done

