echo "" >> results.txt
echo Run tests >> results.txt
echo "" >> results.txt
start_time=$( date +"%s" )
for example in {0..9}
  do
    echo "" >> results.txt
    echo example = $example >> results.txt
    python main.py --example $example >> results.txt
  done

echo Finished! >> results.txt
elapsed_time=$(($( date +"%s" )-$start_time))
echo Elapsed Time: $elapsed_time >> results.txt
