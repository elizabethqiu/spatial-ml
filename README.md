# SpatialStack
Beyond the pixel: Stacking time and depth to give AI a sense of space.

Try it out: www.spatial-ml.tech

Built for Hacktech 2026

## Inspiration
LLMs are really bad at spacial awareness. If you give an LLM an image including an apple and banana, and ask which one is on the right and left, it will be right 50% of the time. This is equivalent to guessing. Since this problem was so open-ended we were inspired to think of the most practical solution we could and implement it. 

## What it does
Our model correctly identifies objects within snapshots of construction videos. By identifying the objects and comparing their distances and orientation geometrically, we are able to make progress on improving spatial orientation reasoning. 

## How we built it
We used the existing pre-trained model from the SpatialVLM paper [link](https://arxiv.org/pdf/2401.12168). 
We then used the 6 egocentric construction videos provided by IronSite and fine-tuned the model on those.

## Challenges we ran into
One challenge we ran into was scale ambiguity. Because the videos are egocentric (filmed through a bodycam on the front of a construction worker), the relative scales throughout the videos varied a lot. This definitely affects how distances match up to real-world value (e.g. how far is the lamp from the fireplace in meters?). By focusing on relative distances (e.g. is the lamp or fireplace closer), we were able to get around some of these problems. 

## Accomplishments that we're proud of
We are proud of achieving cross-domain spatial generalization. The model accurately identified spatial relationships in a construction setting after some fine-tuning, even though the initial model we used has had only previously seen other types of training data. 

## What we learned
Geometry is much more important than labels - for spatial tasks, high-quality depth data is often more important than high-quality text labels.

## What’s next for SpatialStack
We want to deepen the fine-tuning process. Given the brief nature of HackTech, we’ve only scratched the surface of what is possible. We plan to explore more intensive fine-tuning methods, such as LoRA (Low-Rank Adaptation) or QLoRA, to further refine the model’s metric precision.

## Fine-tuning Setup

Before running `finetune.ipynb` in Colab, you need to upload the frames to HuggingFace from your local machine (one-time step):

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token=’YOUR_HF_TOKEN’)
api.upload_large_folder(
    repo_id=’elizqiu/spatial-ml’,
    repo_type=’dataset’,
    folder_path=’data’,
    allow_patterns=’frames/**’,
)
print(‘Done’)
"
```

Then open `finetune.ipynb` in Colab, select an H100 GPU runtime, fill in your HF token in the config cell, and run all cells.

## How to run website locally
```
cd frontend
npm run dev
```
