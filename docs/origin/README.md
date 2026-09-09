# Origin

Two files from the day the project started, in the author's own words, kept unedited.

`gorev.txt` is the one-line brief: train a LoRA adapter on the Kimi K3 model sitting on the
D: drive of this computer so that it speaks Turkish. `plan_ve_gorev.txt` is the plan written
just after it: apply LoRA to a language model without ever holding it in RAM or VRAM, which
requires the model to be a Mixture-of-Experts; move weights between VRAM, RAM and disk under
the control of a purpose-written engine; scan the parameters lazily, layer by layer; and test
before training, because once training starts what is being spent is time.

They are here because the plan turned out to be right about the hard parts, and because a
repository that shows what it set out to do is easier to judge than one that only shows what
it ended up with. The D: drive named in them belonged to a Windows machine that no longer
exists; the work now runs on Linux with the checkpoint on an external USB disk.
