










```sh
code /home/dimitri/code/oll_onemax_visualization/src/visualize.ipynb



# This will run first config via bash.
find config/continuous/ -name '*.yaml' -print0 | xargs -0 -I {} .deploy/RUN_LOCAL {}

# This will run all configs via sbatch.
find config/four_parameters/ -name '*.yaml' -print0 | xargs -0 -I {} .deploy/RUN_SLURM {}

rm -rf /home/dimitri/code/oll_onemax/computed/cirrus-login2/test
rsync -avz --progress cirrus:/work/sc122/sc122/dimitri_rusin/oll_onemax/computed/cirrus-login2/test/ /home/dimitri/code/oll_onemax/computed/cirrus-login2/test
```




Instead of modifying the .gitmodules file directly, run this command:
```sh
git submodule add https://github.com/dimitri-rusin/OLL.git DE0CH_OLL

git submodule deinit -f --all
git submodule init
git submodule update

git submodule add --force https://github.com/dimitri-rusin/OLL.git DE0CH_OLL
```

```sh
git submodule update --init --recursive
```










```sh
.deploy/RUN config/ppo/guy.yaml
.deploy/RUN_SBATCH config/compare_cont_disc/climatic.yaml
```





```sh





mkdir -p /home/dimitri/code/oll_onemax/computed/cirrus-login2/continuous
rsync -avz --progress cirrus:/work/sc122/sc122/dimitri_rusin/oll_onemax/computed/cirrus-login2/continuous/ /home/dimitri/code/oll_onemax/computed/cirrus-login2/continuous




sacct --job=5598911
squeue -u $USER | grep 5598911


```


```sh
# This will run first config via bash.
find config/continuous/ -name '*.yaml' -print0 | xargs -0 -I {} bash .deploy/RUN_LOCAL {}

# This will run all configs via sbatch.
find config/test/ -name '*.yaml' -print0 | xargs -0 -I {} sbatch .deploy/RUN_LOCAL {}
```

Convert .py to .ipynb:
```sh

cd /home/dimitri/code/py2ipynb
conda activate .deploy/conda_environment/
python src/py2ipynb.py /home/dimitri/code/oll_onemax/src/visualize.py /home/dimitri/code/oll_onemax/src/visualize.ipynb

```

```sh




sbatch .deploy/RUN_GENERATE config/compare_cont_disc/batch.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/climatic.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/colony.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/dealmaker.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/emoticon.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/fanciness.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/rubdown.yaml
sbatch .deploy/RUN_GENERATE config/compare_cont_disc/wasabi.yaml

sbatch .deploy/RUN_GENERATE config/ppo/anytime.yaml
sbatch .deploy/RUN_GENERATE config/ppo/area.yaml
sbatch .deploy/RUN_GENERATE config/ppo/disorder.yaml
sbatch .deploy/RUN_GENERATE config/ppo/garage.yaml
sbatch .deploy/RUN_GENERATE config/ppo/giggly.yaml
sbatch .deploy/RUN_GENERATE config/ppo/glacial.yaml
sbatch .deploy/RUN_GENERATE config/ppo/glorified.yaml
sbatch .deploy/RUN_GENERATE config/ppo/handshake.yaml
sbatch .deploy/RUN_GENERATE config/ppo/ladle.yaml
sbatch .deploy/RUN_GENERATE config/ppo/pediatric.yaml
sbatch .deploy/RUN_GENERATE config/ppo/pep.yaml
sbatch .deploy/RUN_GENERATE config/ppo/primp.yaml
sbatch .deploy/RUN_GENERATE config/ppo/propeller.yaml
sbatch .deploy/RUN_GENERATE config/ppo/protector.yaml
sbatch .deploy/RUN_GENERATE config/ppo/quality.yaml
sbatch .deploy/RUN_GENERATE config/ppo/underhand.yaml








sbatch .deploy/RUN_GENERATE config/ppo/anthology.yaml
sbatch .deploy/RUN_GENERATE config/ppo/chance.yaml
sbatch .deploy/RUN_GENERATE config/ppo/drank.yaml
sbatch .deploy/RUN_GENERATE config/ppo/exclude.yaml
sbatch .deploy/RUN_GENERATE config/ppo/gave.yaml
sbatch .deploy/RUN_GENERATE config/ppo/glacier.yaml
sbatch .deploy/RUN_GENERATE config/ppo/guy.yaml
sbatch .deploy/RUN_GENERATE config/ppo/handheld.yaml
sbatch .deploy/RUN_GENERATE config/ppo/mounted.yaml
sbatch .deploy/RUN_GENERATE config/ppo/pecan.yaml
sbatch .deploy/RUN_GENERATE config/ppo/penniless.yaml
sbatch .deploy/RUN_GENERATE config/ppo/plural.yaml
sbatch .deploy/RUN_GENERATE config/ppo/preschool.yaml
sbatch .deploy/RUN_GENERATE config/ppo/racing.yaml
sbatch .deploy/RUN_GENERATE config/ppo/sublease.yaml
sbatch .deploy/RUN_GENERATE config/ppo/timothy.yaml



OO__DB_PATH=computed/data/shadily.db python src/visualize.py --port 9000
OO__DB_PATH=computed/cirrus/compare_cont_disc/batch.db python src/visualize.py --port 9000
OO__DB_PATH=computed/cirrus/compare_cont_disc/climatic.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/compare_cont_disc/colony.db python src/visualize.py --port 9002
OO__DB_PATH=computed/cirrus/compare_cont_disc/dealmaker.db python src/visualize.py --port 9003
OO__DB_PATH=computed/cirrus/compare_cont_disc/emoticon.db python src/visualize.py --port 9004
OO__DB_PATH=computed/cirrus/compare_cont_disc/fanciness.db python src/visualize.py --port 9005
OO__DB_PATH=computed/cirrus/compare_cont_disc/rubdown.db python src/visualize.py --port 9006
OO__DB_PATH=computed/cirrus/compare_cont_disc/wasabi.db python src/visualize.py --port 9007



OO__DB_PATH=computed/cirrus/anthology.db python src/visualize.py --port 9000 # stabil ab 84,000
OO__DB_PATH=computed/cirrus/chance.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/drank.db python src/visualize.py --port 9002 # stabil ab 40,000
OO__DB_PATH=computed/cirrus/exclude.db python src/visualize.py --port 9003 # stabil ab 4,000
OO__DB_PATH=computed/cirrus/gave.db python src/visualize.py --port 9004
OO__DB_PATH=computed/cirrus/glacier.db python src/visualize.py --port 9005
OO__DB_PATH=computed/cirrus/guy.db python src/visualize.py --port 9006
OO__DB_PATH=computed/cirrus/handheld.db python src/visualize.py --port 9007
OO__DB_PATH=computed/cirrus/mounted.db python src/visualize.py --port 9008
OO__DB_PATH=computed/cirrus/pecan.db python src/visualize.py --port 9009
OO__DB_PATH=computed/cirrus/penniless.db python src/visualize.py --port 9010
OO__DB_PATH=computed/cirrus/plural.db python src/visualize.py --port 9011 # stabil ab 96,000
OO__DB_PATH=computed/cirrus/preschool.db python src/visualize.py --port 9012
OO__DB_PATH=computed/cirrus/racing.db python src/visualize.py --port 9013
OO__DB_PATH=computed/cirrus/sublease.db python src/visualize.py --port 9014
OO__DB_PATH=computed/cirrus/timothy.db python src/visualize.py --port 9015











.deploy/RUN_GENERATE config/ppo/anthology.yaml &
.deploy/RUN_GENERATE config/ppo/chance.yaml &
.deploy/RUN_GENERATE config/ppo/drank.yaml &
.deploy/RUN_GENERATE config/ppo/exclude.yaml &
.deploy/RUN_GENERATE config/ppo/gave.yaml &
.deploy/RUN_GENERATE config/ppo/glacier.yaml &
.deploy/RUN_GENERATE config/ppo/guy.yaml &
.deploy/RUN_GENERATE config/ppo/handheld.yaml &
.deploy/RUN_GENERATE config/ppo/mounted.yaml &
.deploy/RUN_GENERATE config/ppo/pecan.yaml &
.deploy/RUN_GENERATE config/ppo/penniless.yaml &
.deploy/RUN_GENERATE config/ppo/plural.yaml &
.deploy/RUN_GENERATE config/ppo/preschool.yaml &
.deploy/RUN_GENERATE config/ppo/racing.yaml &
.deploy/RUN_GENERATE config/ppo/sublease.yaml &
.deploy/RUN_GENERATE config/ppo/timothy.yaml &


/home/dimitri/code/oll_onemax/config/ppo/chance.yaml
/home/dimitri/code/oll_onemax/config/ppo/drank.yaml
/home/dimitri/code/oll_onemax/config/ppo/exclude.yaml
/home/dimitri/code/oll_onemax/config/ppo/gave.yaml
/home/dimitri/code/oll_onemax/config/ppo/glacier.yaml
/home/dimitri/code/oll_onemax/config/ppo/guy.yaml
/home/dimitri/code/oll_onemax/config/ppo/handheld.yaml
/home/dimitri/code/oll_onemax/config/ppo/mounted.yaml
/home/dimitri/code/oll_onemax/config/ppo/pecan.yaml
/home/dimitri/code/oll_onemax/config/ppo/penniless.yaml
/home/dimitri/code/oll_onemax/config/ppo/plural.yaml
/home/dimitri/code/oll_onemax/config/ppo/preschool.yaml
/home/dimitri/code/oll_onemax/config/ppo/racing.yaml
/home/dimitri/code/oll_onemax/config/ppo/sublease.yaml
/home/dimitri/code/oll_onemax/config/ppo/timothy.yaml
```




```sh
source (python .deploy/apply.py config/ppo.yaml | psub)
python src/generate.py

OO__DB_PATH=computed/data/humble.db python src/visualize.py --port 9001
```








```sh
OO__DB_PATH=computed/cirrus/bobbing.db python src/visualize.py --port 9001 # changes a lot, still way above baseline
OO__DB_PATH=computed/cirrus/calamari.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/celibacy.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/denatured.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/glacial.db python src/visualize.py --port 9001 # comes close to baseline, worsens over time
OO__DB_PATH=computed/cirrus/gotten.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/iphone.db python src/visualize.py --port 9001 # also dips down, becomes worse
OO__DB_PATH=computed/cirrus/laborer.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/moonscape.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/nerd.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/plunging.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/retention.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/send.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/shorts.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/vocalist.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/voyage.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/waltz.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/willed.db python src/visualize.py --port 9001
```


There seems to be a problem when a generate python script creates the data/ directory, or any directory, we get:
```
Matplotlib created a temporary config/cache directory at /dev/shm/dimitri_rusin_5551730/matplotlib-sd8h8zr6 because the default path (/home/sc122/sc122/dimitri_rusin/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Fontconfig error: No writable cache directories
Traceback (most recent call last):
  File "/mnt/lustre/e1000/home/sc122/sc122/dimitri_rusin/oll_onemax/src/generate.py", line 476, in <module>
    main()
  File "/mnt/lustre/e1000/home/sc122/sc122/dimitri_rusin/oll_onemax/src/generate.py", line 343, in main
    database = setup_database(config['db_path'])
  File "/mnt/lustre/e1000/home/sc122/sc122/dimitri_rusin/oll_onemax/src/generate.py", line 471, in setup_database
    os.makedirs(directory_path)
  File "/mnt/lustre/e1000/home/sc122/sc122/dimitri_rusin/oll_onemax/.deploy/conda_environment/lib/python3.10/os.py", line 225, in makedirs
    mkdir(name, mode)
FileExistsError: [Errno 17] File exists: 'computed/data'
```
So, maybe it's better to create all directories in shell, instead of python which is run in multiple processes in parallel.
With the computed/data/ directory already existing before the sbatch commands below are launched, no slurm output file contains a message like the above one.




```sh
sbatch .deploy/generate.slurm config/ppo/bobbing.yaml
sbatch .deploy/generate.slurm config/ppo/calamari.yaml
sbatch .deploy/generate.slurm config/ppo/celibacy.yaml
sbatch .deploy/generate.slurm config/ppo/denatured.yaml
sbatch .deploy/generate.slurm config/ppo/glacial.yaml
sbatch .deploy/generate.slurm config/ppo/gotten.yaml
sbatch .deploy/generate.slurm config/ppo/iphone.yaml
sbatch .deploy/generate.slurm config/ppo/laborer.yaml
sbatch .deploy/generate.slurm config/ppo/moonscape.yaml
sbatch .deploy/generate.slurm config/ppo/nerd.yaml
sbatch .deploy/generate.slurm config/ppo/plunging.yaml
sbatch .deploy/generate.slurm config/ppo/retention.yaml
sbatch .deploy/generate.slurm config/ppo/send.yaml
sbatch .deploy/generate.slurm config/ppo/shorts.yaml
sbatch .deploy/generate.slurm config/ppo/vocalist.yaml
sbatch .deploy/generate.slurm config/ppo/voyage.yaml
sbatch .deploy/generate.slurm config/ppo/waltz.yaml
sbatch .deploy/generate.slurm config/ppo/willed.yaml


bash .deploy/generate.slurm config/ppo/gotten.yaml
```



```sh
source <(python .deploy/apply.py config/ppo.yaml)


source (python .deploy/apply.py config/ppo.yaml | psub)

source (python .deploy/apply.py config/ppo/celibacy.yaml | psub)
python src/generate.py

OO__DB_PATH=computed/data/ppo.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/ppo.db python src/visualize.py --port 9001


bash .deploy/generate.slurm config/ppo.yaml
. .deploy/generate.slurm config/ppo.yaml
sbatch .deploy/generate.slurm config/ppo.yaml
```








```sh
rsync -avz --progress cirrus:/work/sc122/sc122/dimitri_rusin/oll_onemax/computed/data/ /home/dimitri/code/oll_onemax/computed/cirrus



OO__DB_PATH=computed/cirrus_without_sqrt_dimensions_average_out_good_solutions/dim_500.db python src/visualize.py --port 9001
OO__DB_PATH=computed/data/attribute.db python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/gotten.yaml python src/visualize.py --port 9001
OO__DB_PATH=computed/cirrus/glacial.db python src/visualize.py --port 9001

```

```
computed/data/untangled_shy_unread_dragster_scam.db

OO__EXECUTION__EPISODE_ID_LOW=1 OO__EXECUTION__EPISODE_ID_HIGH=9999 OO__DB_PATH=computed/cirrus/untangled_shy_unread_dragster_scam.db OO__N=50 python tests/test_evaluation_episodes_with_rust.py
```

Visualize experiment runs:
```sh
conda activate .deploy/conda_environment/
screen -ls | grep 'Detached' | awk '{print $1}' | xargs -I {} screen -S {} -X quit
ls computed/cirrus | nl -v 8061 | xargs -L 1 sh -c 'filename=$(basename $1 .db); screen -dmS ${filename}__$0 fish -c "conda activate .deploy/conda_environment; and OO__DB_PATH=computed/cirrus/$1 python src/visualize.py --port $0"'
OO__DB_PATH=computed/cirrus python src/index.py --port 8052
```

```sh

ls computed/cirrus | nl -v 8061 | xargs -L 1 sh -c 'filename=$(basename $1 .db); screen -dmS ${filename}__$0 fish -c "conda activate .deploy/conda_environment; and OO__DB_PATH=computed/cirrus/$1 python src/visualize.py --port $0"'

screen -ls | grep 'Detached' | awk '{print $1}' | xargs -I {} screen -S {} -X quit

screen -ls

```



```
bash .deploy/generate.slurm config/a_few_more_configs_March_17/abroad.yaml

sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/abroad.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/activate.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/amused.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/angles.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/another.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/anthology.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/asleep.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/astrology.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/bats.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/bazooka.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/bubbly.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/buddy.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/camcorder.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/caring.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/caterer.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/cement.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/chief.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/clover.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/composer.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/confiding.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/conical.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/coronary.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/deceased.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/deck.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/demeaning.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/dial.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/dove.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/draw.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/dumping.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/earmuff.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/elaborate.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/ending.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/explode.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/exterior.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/feast.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/tip.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/flagstone.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/follow.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/freebase.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/gallantly.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/gigabyte.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/glowworm.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/handwash.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/hardcover.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/hurry.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/jasmine.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/jingle.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/juggle.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/jugular.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/keenly.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/lilac.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/matron.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/multiply.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/mummy.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/mutilator.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/nebula.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/overbuilt.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/overcrowd.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/pauper.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/pellet.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/petty.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/precook.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/protector.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/rearrange.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/retiring.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/revise.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/ridden.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/rind.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/scorecard.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/showing.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/sibling.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/skinny.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/sliver.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/smith.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/snitch.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/stand.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/starter.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/stoke.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/strut.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/superior.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/surrogate.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/theatrics.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/tidbit.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/trading.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/trimester.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/undefined.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/unpeeled.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/untangled.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/untidy.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/upload.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/variety.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/washable.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/washcloth.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/whacky.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/yapping.yaml
sbatch .deploy/generate.slurm config/a_few_more_configs_March_17/zesty.yaml
```



```sh
sbatch .deploy/generate.slurm config/March_11/attribute_gathering_uplifting_chump_kudos.yaml
sbatch .deploy/generate.slurm config/March_11/bats_unlawful_crate_retype_sedative.yaml
sbatch .deploy/generate.slurm config/March_11/bubbly_disfigure_flying_scion_girdle.yaml
sbatch .deploy/generate.slurm config/March_11/camcorder_enticing_claw_slot_outburst.yaml
sbatch .deploy/generate.slurm config/March_11/construct_handclap_pushover_overeager_runway.yaml
sbatch .deploy/generate.slurm config/March_11/earmuff_overlay_grass_scrabble_procedure.yaml
sbatch .deploy/generate.slurm config/March_11/exterior_wrought_flanked_diabetes_sudden.yaml
sbatch .deploy/generate.slurm config/March_11/freebase_alike_upcoming_pummel_resilient.yaml
sbatch .deploy/generate.slurm config/March_11/glaring_banish_nursery_hungrily_spooky.yaml
sbatch .deploy/generate.slurm config/March_11/handwash_bonehead_scarring_duress_morphine.yaml
sbatch .deploy/generate.slurm config/March_11/hurry_filing_raft_uneven_sierra.yaml
sbatch .deploy/generate.slurm config/March_11/jugular_starter_padlock_herself_cosponsor.yaml
sbatch .deploy/generate.slurm config/March_11/mutilator_smolder_flypaper_overhead_happiness.yaml
sbatch .deploy/generate.slurm config/March_11/showing_rigor_certainty_drainable_strongman.yaml
sbatch .deploy/generate.slurm config/March_11/spooky_grazing_dole_filter_opossum.yaml
sbatch .deploy/generate.slurm config/March_11/superior_container_headband_majority_bunkmate.yaml
sbatch .deploy/generate.slurm config/March_11/tackiness_circle_engraved_axis_wisdom.yaml
sbatch .deploy/generate.slurm config/March_11/theme_yin_tartar_upstairs_pancake.yaml
sbatch .deploy/generate.slurm config/March_11/tidbit_rudder_undying_recount_bronze.yaml
sbatch .deploy/generate.slurm config/March_11/uncertain_avenue_deftly_outback_winking.yaml
sbatch .deploy/generate.slurm config/March_11/untangled_shy_unread_dragster_scam.yaml
sbatch .deploy/generate.slurm config/March_11/upload_antiquity_linguini_headpiece_glutinous.yaml
sbatch .deploy/generate.slurm config/March_11/variety_poise_exes_animator_constrict.yaml
sbatch .deploy/generate.slurm config/March_11/washable_underhand_poster_magnesium_drastic.yaml
```


```sh
python .deploy/range.py
source (python .deploy/apply.py config/March_11/attribute_gathering_uplifting_chump_kudos.yaml | psub)
python src/generate.py


source (python .deploy/apply.py config/a_few_more_configs_March_17/zesty.yaml | psub)
python src/generate.py


```

```sh
bash .deploy/generate.slurm config/March_11/affection_directive_tavern_ruined_degrease.yaml

sbatch .deploy/generate.slurm config/dim_50.yaml
sbatch .deploy/generate.slurm config/dim_500.yaml
sbatch .deploy/generate.slurm config/dim_1_000.yaml


sacct --job=5531370
sacct --job=5531371
sacct --job=5531372



OO__DB_PATH=./computed/cirrus/March_17_07h_39m_26s__dim_50.db python src/visualize.py
OO__DB_PATH=computed/data/attribute_gathering_uplifting_chump_kudos.db python src/visualize.py
OO__DB_PATH=computed/cirrus/attribute_gathering_uplifting_chump_kudos.db python src/visualize.py
```

```


OO__DB_PATH=computed/cirrus/attribute_gathering_uplifting_chump_kudos.db python src/visualize.py --port 8050 &
OO__DB_PATH=computed/cirrus/bats_unlawful_crate_retype_sedative.db python src/visualize.py --port 8051 &
OO__DB_PATH=computed/cirrus/bubbly_disfigure_flying_scion_girdle.db python src/visualize.py --port 8052 &
OO__DB_PATH=computed/cirrus/camcorder_enticing_claw_slot_outburst.db python src/visualize.py --port 8053 &

```


```sh
module load anaconda3/2023.09
./.deploy/BUILD
./.deploy/RUN

squeue -u $USER
```






```sh

OO__DB_PATH=./computed/cirrus/dim_50.db python src/visualize.py

OO__EXECUTION__EPISODE_ID_LOW=1 OO__EXECUTION__EPISODE_ID_HIGH=9999 OO__DB_PATH=./computed/cirrus/dim_50.db OO__N=50 python tests/test_evaluation_episodes_with_rust.py




OO__EXECUTION__EPISODE_ID_LOW=1 OO__EXECUTION__EPISODE_ID_HIGH=9999 OO__DB_PATH=/home/dimitri/code/oll_onemax/computed/data/March_15_17h_39m_14s__dim_50.db OO__N=50 python tests/test_evaluation_episodes_with_rust.py

OO__EXECUTION__EPISODE_ID_LOW=1 OO__EXECUTION__EPISODE_ID_HIGH=9999 OO__DB_PATH=/home/dimitri/code/oll_onemax/computed/data/March_15_10h_31m_28s__dim_50.db OO__N=50 python tests/test_evaluation_episodes_with_rust.py

OO__EXECUTION__EPISODE_ID_LOW=9 OO__EXECUTION__EPISODE_ID_HIGH=13 OO__DB_PATH=/home/dimitri/code/oll_onemax/computed/data/March_15_10h_31m_28s__dim_50.db OO__N=50 python tests/test_evaluation_episodes_with_rust.py


OO_DB_PATH=./computed/cirrus/dim_50.db python visual.py
```



```
[dimitri_rusin@cirrus-login1 oll_onemax]$ sbatch strategies.slurm dim_50.yaml
Submitted batch job 5524237
[dimitri_rusin@cirrus-login1 oll_onemax]$ sbatch strategies.slurm dim_500.yaml
Submitted batch job 5524238
[dimitri_rusin@cirrus-login1 oll_onemax]$ sbatch strategies.slurm dim_1_000.yaml
Submitted batch job 5524239
[dimitri_rusin@cirrus-login1 oll_onemax]$
```




```sh
ssh cirrus



module load anaconda3/2023.09
git pull
conda_path=$(which conda)
if [[ "$conda_path" == */condabin/conda ]]; then
  conda_sh_path="${conda_path%/condabin/conda}/etc/profile.d/conda.sh"
elif [[ "$conda_path" == */bin/conda ]]; then
  conda_sh_path="${conda_path%/bin/conda}/etc/profile.d/conda.sh"
else
  echo "Error: Unable to locate the conda.sh file."
  exit 1
fi
echo "Sourcing: $conda_sh_path"
source "$conda_sh_path"

bash strategies.slurm dim_50.yaml
```



```sh
OO_DB_PATH=./cirrus/dim_50.db python visual.py
OO_DB_PATH=./cirrus/dim_500.db python visual.py
OO_DB_PATH=./cirrus/dim_1_000.db python visual.py
```

```sh
rsync -avz --progress cirrus:/work/sc122/sc122/dimitri_rusin/oll_onemax/data/ /home/dimitri/code/oll_onemax/cirrus
```

```sh
source (python .deploy/apply_setting.py ./experiment_settings/another.yaml | psub); and python src/generate.py

source (python yaml2env.py dim_50.yaml | psub); and python generate_policies.py
source (python yaml2env.py dim_50.yaml | psub); and python visual.py

source (python yaml2env.py .env.yaml | psub); and python visual.py
```



```sh
-> sbatch strategies.slurm dim_50.yaml
  Submitted batch job 5522562
-> sbatch strategies.slurm dim_500.yaml
  Submitted batch job 5522563
-> sbatch strategies.slurm dim_1_000.yaml
  Submitted batch job 5522564

sacct --job=5522562
sacct --job=5522563
sacct --job=5522564

sacct --job=5523180
sacct --job=5523181
sacct --job=5523182


sacct --job=5523187
sacct --job=5523188
sacct --job=5523189


```




It's not possible to export an environment variable after activating an Anaconda3 environment.

```sh
source (python yaml2env.py oll_onemax.yaml --clean | psub)

# Fish Shell
source (python yaml2env.py oll_onemax.yaml | psub)
source (python yaml2env.py oll_onemax.yaml --clean | psub)

# Bash Shell
source <(python yaml2env.py oll_onemax.yaml)
source <(python yaml2env.py oll_onemax.yaml --clean)

# Zsh Shell
source =(python yaml2env.py oll_onemax.yaml)
source =(python yaml2env.py oll_onemax.yaml --clean)
```




Commands for Cirrus:
```sh
sbatch strategies.slurm
squeue --job=5507492
sacct --job=5507492
sacct --job=5508417
sacct --job=5507956
sacct --job=5510269


sacct --job=5522562
sacct --job=5522563
sacct --job=5522564



scp cirrus:/mnt/lustre/indy2lfs/work/sc122/sc122/dimitri_rusin/oll_onemax/data/policies.db /home/dimitri/code/oll_onemax/cirrus/



sacct --job=5520125


rsync -avz --progress cirrus:/work/sc122/sc122/dimitri_rusin/oll_onemax/data/policies.db /home/dimitri/code/oll_onemax/cirrus/policies.db
```

```sh
conda_path=$(which conda)
if [[ "$conda_path" == */condabin/conda ]]; then
  conda_sh_path="${conda_path%/condabin/conda}/etc/profile.d/conda.sh"
elif [[ "$conda_path" == */bin/conda ]]; then
  conda_sh_path="${conda_path%/bin/conda}/etc/profile.d/conda.sh"
else
  echo "Error: Unable to locate the conda.sh file."
  exit 1
fi
echo "Sourcing: $conda_sh_path"
source "$conda_sh_path"
```



If you want to pull a force-pushed branch.
```
git config pull.rebase true
```

```sh
ssh cirrus
module load anaconda3/2023.09

cd /work/sc122/sc122/dimitri_rusin/oll_onemax/
# cd /mnt/lustre/indy2lfs/work/sc122/sc122/dimitri_rusin/
# rm -rf oll_onemax/
git clone https://github.com/dimitri-rusin/oll_onemax.git
cd oll_onemax/

git submodule init
git submodule update

rm -rf ./.conda_environment/
rm -rf ./paper_code/onell_algs_rs/target/
conda env create --prefix ./.conda_environment/ --file .conda.yaml
conda activate ./.conda_environment/
pip install --requirement .pip.txt

(
  cd ./paper_code/onell_algs_rs/
  maturin build --release
)
find ./paper_code/onell_algs_rs/target/wheels -name "*.whl" -print0 | xargs -0 pip install
```





On cirrus, explicitly run bash before activating any conda environment:
```
ssh cirrus
bash
conda activate base
```





Falls .gitmodules nicht funktioniert:
```sh
git clone https://github.com/automl/DACBench.git dacbench
```


```sh
module load anaconda/python3
```


```sh
ssh mesu
cd /scratchbeta/rusind/oll
conda activate base
rm -rf ./.conda_environment/
rm -rf ./paper_code/onell_algs_rs/target/
conda env create --prefix ./.conda_environment/ --file .conda.yaml
conda activate ./.conda_environment/
pip install --requirement .pip.txt

cd ./paper_code/onell_algs_rs/
maturin build --release
cd -
find ./paper_code/onell_algs_rs/target/wheels -name "*.whl" -print0 | xargs -0 pip install --force-reinstall

git submodule init
git submodule update
```











Install a Rust-implemented python module
```sh
cd /home/dimitri/code/oll_onemax/paper_code/onell_algs_rs/
pip install --upgrade maturin===0.13.7
maturin build --release
pip install --force-reinstall target/wheels/onell_algs_rs-0.1.0-cp310-cp310-manylinux_2_31_x86_64.whl
```

```py
import onell_algs_rs
result = onell_algs_rs.onell_lambda(n, lbds, seed, max_evals)
```



We get messages like:
```sh
=>> PBS: job killed: walltime 79 exceeded limit 60
```


Run an experiment on the MeSU supercomputer:
```sh
ssh mesu
cd /scratchbeta/rusind/oll_onemax/
git fetch
git checkout -- .
git pull
qsub mesu.bash > mesu.job

cd /scratchbeta/rusind/oll/
git clone https://github.com/dimitri-rusin/oll_onemax.git oll
cd oll
git submodule init
git submodule update


# Read execution info of MeSU job.
qstat $(cat mesu.job)
qstat $(cat mesu.job) -x

# Read output of MeSu job.
cat "onemaxoll.o$(cat mesu.job | cut -d'.' -f1)"

# Remove MeSU job out of the queue.
qdel $(cat mesu.job)

rm -rf /home/dimitri/code/oll_onemax/mesu/policies.db
mkdir /home/dimitri/code/oll_onemax/mesu/
scp mesu:/scratchbeta/rusind/oll_onemax/data/policies.db /home/dimitri/code/oll_onemax/mesu/policies.db
```


Get the mesu prepared stuff to my local system for visualization.
```sh
rm -rf /home/dimitri/code/oll_onemax/mesu/policies.db
scp mesu:/scratchbeta/rusind/oll_onemax/data/policies.db /home/dimitri/code/oll_onemax/mesu/policies.db
```





```sh
#!/usr/bin/env bash

#PBS -q beta
#PBS -l select=1:ncpus=24
#PBS -l walltime=01:00:00
#PBS -N onemaxoll
#PBS -j oe

module add cmake/3.22
module add conda3-2023.02
module add gcc/11.2
module add git/2.42.0
module add LLVM/clang-llvm-10.0

export PATH="/scratchbeta/rusind/oll_onemax/.conda_environment/bin:$PATH"

python /scratchbeta/rusind/oll_onemax/strategies.py
```






```sh
# Job id            Name             User              Time Use S Queue
# ----------------  ---------------- ----------------  -------- - -----
# 2081359.mesu2     onemaxoll        rusind            00:00:02 R b_normal

qsub mesu.bash
qstat 2081359.mesu2
qstat 2081359.mesu2 -x
qdel 2085650.mesu2
```



```sh
#!/usr/bin/env bash

#PBS -q beta
#PBS -l select=1:ncpus=24
#PBS -l walltime=00:01:00
#PBS -N onemaxoll
#PBS -j oe

module add cmake/3.22
module add conda3-2023.02
module add gcc/11.2
module add git/2.42.0
module add LLVM/clang-llvm-10.0

conda init
conda activate /scratchbeta/rusind/oll_onemax/.conda_environment/
python /scratchbeta/rusind/oll_onemax/strategies.py
```



```sh
#!/usr/bin/env bash

#PBS -q beta
#PBS -l select=1:ncpus=24
#PBS -l walltime=00:01:00
#PBS -N onemaxoll
#PBS -j oe

module add cmake/3.22
module add conda3-2023.02
module add gcc/11.2
module add git/2.42.0
module add LLVM/clang-llvm-10.0

export PATH="/scratchbeta/rusind/oll_onemax/.conda_environment/bin:$PATH"

python /scratchbeta/rusind/oll_onemax/strategies.py
```




















```sh
ssh mesu

module add cmake/3.22
module add conda3-2023.02
module add gcc/11.2
module add git/2.42.0
module add LLVM/clang-llvm-10.0



cd /scratchbeta/rusind/
rm -rf oll_onemax/
git clone --branch tabular --depth 1 git@github.com:dimitri-rusin/oll_onemax.git
cd oll_onemax
git submodule init
git submodule update

conda activate base
rm -rf ./.conda_environment/
conda env create --prefix ./.conda_environment/ --file .conda.yaml
conda activate ./.conda_environment/
pip install --requirement .pip.txt


```













```sh
ssh mesu

cd francois/
./RUN

ll /scratchbeta/rusind/
cat /scratchbeta/rusind/4c_20_LogFile.txt
cat /scratchbeta/rusind/4c_20.jl

qsub RUN

qstat 2006960.mesu2
qstat 2006960.mesu2 -x
qdel 2006960.mesu2

rm -rf /scratchbeta/rusind/4c_19*

scp mesu:/scratchbeta/rusind/4c_20_LogFile.txt .
scp mesu:/scratchbeta/rusind/4c_20.jl .
```


















Install a Rust-implemented python module
```sh
pip install maturin
maturin build --release
pip install target/wheels/onell_algs_rs-0.1.0-cp310-cp310-manylinux_2_31_x86_64.whl

import onell_algs_rs
result = onell_algs_rs.onell_lambda(n, lbds, seed, max_evals)
```




This should be the step function of the environment:
```py
# mutation phase
p = lambda_ / self.num_dimensions
xprime, f_xprime, ne1 = x.mutate(p, lambda_, rng)

# crossover phase
c = 1 / lambda_
y, f_y, ne2 = x.crossover(xprime, c, lambda_, include_xprime_crossover, count_different_inds_only,  rng)

# selection phase
old_f_x = f_x
if f_x <= f_y:
  x = y
  f_x = f_y
```










NGUYEN INFO
==============================================================================

I have an example script for doing the plotting (the learning curve plot and the policy plot) for the LeadingOnes benchmark here: https://github.com/ndangtt/LeadingOnesDACNew/blob/main/examples/analyse_results.py

You can try that script by following example 2:

https://github.com/ndangtt/LeadingOnesDACNew/tree/main

Here is our FOGA paper published year on using irace to control the parameter of the (1+lambda,lambda)-GA on OneMax problem:

https://arxiv.org/abs/2302.12334

Here is the algorithm in Python:

https://github.com/DE0CH/OLL/blob/ceeb3b118291cc72bfe3a40c1577983bf487ac41/tuned-with-irace/onell_algs.py#L445

(the lbds parameter is the policy, it's an array of size n, where the i^th element tells us which lambda value we should use for fitness value of i)

Deyao (my student) also reimplemented this algorithm in Rust. It significantly reduces the compute time, we can use this one for the evaluation (while using the Python code for the training):

https://github.com/DE0CH/OLL/tree/ceeb3b118291cc72bfe3a40c1577983bf487ac41/onell_algs_rs


It was very nice to meet you yesterday. Here is the SAT solving paper I mentioned yesterday:

https://arxiv.org/abs/2211.12581

Another thing during our conversion was the topic that Carola, Martin, and our collaborators in Freiburg and I are working on at the moment is Dynamic Algorithm Configuration (DAC). We focuses on developing new DAC benchmarks and using them to gain understanding the strengths and weakness of current deep-RL methods in the context of DAC. I thought I'd share our GECCO 2022 here with you as well, just in case you might be interested:

Paper: https://arxiv.org/abs/2202.03259
Blog post: https://andrebiedenkapp.github.io/blog/2022/gecco/
GECCO presentation video: https://www.youtube.com/watch?v=xgljDu5qE-w



February 6, 2024
==============================================================================

So, I just generated a bunch of .zip files. I just want one policy. Can I get that right? Just one policy. Out of this whole thing. Just one policy, essentially save that one policy. Load it, and evaluate for every fitness value, please. Is it that hard?

I still want to generate a policy. Using the PPO project. But now I am handling the outputs. Just, the environment should work properly. Of course. So, I want it to work properly. Just be deterministin on the same input of the random seed. Hmm, let me check that. What are the outputs of the two result functions?

February 7, 2024
==============================================================================

We still could not visualize a single policy in this bitch. We just want a policy: gets a fitness, spits out a lambda. That's all. This should come OUT of using stable_baselines3.PPO. This is ALL that I am asking for.

This function `make_vec_env` actually calls reset with some uncontrolled number as the random seed. This messes up reproducability. So, we work without this function for now.

One could use an actual SQLite database here to save the fitnesses and corresponding lambdas. But anyway, one just needs the different policies. They should be rewritten down. Almost no real need to visualize. Except maybe with a bar chart. Why not, even 100 fitnesses, would be fine. Which is equivalent with having 100 dimensions. Then, all we really need to add is just an actual improvement of the policy. Over time, for each policy, we should probably evaluate it. The "Tensorboard" software could be used, but probably not. But, in any case, it's better to have all the data, in a static way, in a real way, right here on disk. So, the main goal in the end, is of course, to use the real OneMax plus OLL environment and algorithm and at the same time achieve a sequence of constantly improving policies over time, just over time, so basically just a sequence of these. We want to get them. To have them written down. To have them written down right here, right now. Maybe, this won't be too hard. You know, just write it down. Just write it. And then, this is it. Actually. Just this: real environment. Plus ever improving policies.

Just to re-create the env.

Was machen wir mit Nguyen? Wir wollen ein Tensorboard mit den Policies, ganz klar. Ganz klar. Für jeden Step, DEN wir mappen, wollen wir die Policy. Die exakte. Von dem Timestep, von dem Zeitpunkt, und keinem anderen. Das war's eigentlich. Und dann wollen wir die Learning Curve. Wie kriegen wir die Kurve hin? Die Kurve.

Die Kurve ist einfach nur die Evaluation. Die Evaluation der tatsächlichen Policy zu dem konkreten Zeitpunkt. Es gibt mehrere Zeitpunkte: Funktions-Evaluationen, Training minibatch consumptions, timesteps, episodes. Man könnte beliebiges davon als Zeitpunkt nehmen. Oder sogar: Sekunde. Wallclock. Oder auch CPU-time. Ja, im Endeffekt, das sind vielleicht sogar fast alle Möglichkeiten. Also, dann, zu diesem Zeitpunkt: die Policy plus Evaluation. Die Policy-Evaluation geht über eine feste Anzahl von Episoden, bis zu dem Moment, wo wir den vollständigen Return bekommen. Dann average. Das ist die Policy-Evaluation.

Nguyen's job: all we really want is some policies.

Nguyen's job: Let's evaluate the policies. Die Policies müssen alle geradlinig sein. Jede Policy muss rein vor einem erscheinen, so wie die Policy wirklich ist. In Wahrheit. Gott, offenbare mir die Policy. Natürlich das aller letzte Ziel überhaupt ist ein Journal-Paper. Eine Erweiterung, ansetzend, angreifend. Wir wollen weiter veröffentlichen. In das Loch. In das Ziel. Herein. Hinein-kippend. Das Paper... wird nicht nur diese Technik beinhalten. Nein, nein. Wir wollen PPO ausprobieren, aber nur weil Nguyen das so sagt. Die eigentliche Wahrheit ist: wir wollen neural MCTS ausprobieren. Wird das klappen? Wir wollen NRPA ausprobieren. Wird das klappen? Eine Policy wird developed. Eine Policy, welche nicht und überhaupt nicht vom Zustand abhängt, ganz genau so wie RLS_1 und RLS_5. Bitflip, crossover. Alles das - ist möglich - mit NRPA. Mit NRPA Policy Adaptation. Very simply so. Has it been tried? Who tried it so. Then? What came out of it?

Das ist nur die Sekunde, dies ist nur der Moment. Wann wird es festgelegt? Aber später diese andere. Ah.

With NRPA, we try MCTS, too. It's very simple. But it has to be generated. Generated, and properly so.

But for now, PPO will be tried. You just have to plug it in at the correct position. Log the policies. Evaluate them. That's enough. Make a config file. Keep it simple. For now. Thank you.




Running Rust on MeSU.
Problem: Install a .whl file with pip.
=====================

Absolutely, let's include the installation of `setuptools_rust` in the process, as it's required for building Python extensions in Rust. Here's the revised set of instructions:

### 1. Install Development Tools
Install necessary development tools and libraries:
```bash
yum groupinstall -y "Development Tools"
yum install -y openssl-devel bzip2-devel libffi-devel zlib-devel
```

### 2. Install Python 3.6 and pip
Install Python 3.6 and its pip:
```bash
yum install -y python36 python36-pip
```

### 3. Install `tomli` and `setuptools_rust`
Install the `tomli` and `setuptools_rust` Python packages:
```bash
python3.6 -m pip install tomli setuptools_rust
```

### 4. Install Rust
Install Rust using the Rustup script:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 5. Update `Cargo.toml` for Rust Project
In your Rust project, update `Cargo.toml` to use a compatible version of `pyo3`:
```toml
[dependencies]
pyo3 = { version = "0.13", features = ["extension-module"] }
# ... other dependencies remain the same
```

### 6. Build Your Project with Maturin
Install `maturin` and build your Rust project:
```bash
python3.6 -m pip install maturin
maturin build --release -i python3.6
```

### 7. Copy the Built Wheel to the Host
After building the wheel inside the Docker container, copy it to your host machine:
```bash
docker cp <container_id>:/path/to/project/target/wheels /local/path
```
Replace `<container_id>` with your Docker container's ID, `/path/to/project` with the path to your project inside the container, and `/local/path` with the path on your host machine where you want to copy the wheel.

### 8. Install the Wheel on Your System
Install the wheel on your target system using pip:
```bash
pip install /local/path/onell_algs_rs-0.1.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
Replace the wheel filename with the actual filename of the wheel you built.

This comprehensive process covers setting up the CentOS 7 environment with Python 3.6, installing all necessary dependencies, including `tomli` and `setuptools_rust`, and building your Rust project with Python bindings. Ensure to adjust file paths and project names as per your specific setup.

=====================
