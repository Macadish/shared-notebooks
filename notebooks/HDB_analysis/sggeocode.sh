start=(289980 397806 489981 579992 656291 799985 888889 978201)
end=(300000 400000 500000 600000 700000 800000 900000 1000000)
for i in ${!start[@]}; do
    python sggeocode.py ${start[$i]} ${end[$i]} >> sggeocode_"$i"a.csv &
done
