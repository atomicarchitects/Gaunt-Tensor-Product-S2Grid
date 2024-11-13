# Run command
python3 ./MACE/scripts/run_train.py \
    --dataset="3bpa" \
    --subset="train_300K" \
    --default_dtype="float32"\
    --seed=6 \
    --model="EquivariantRealScaleShiftNonLinearBodyOrderedModel" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --device=cuda \
    --max_num_epochs=2000 \
    --patience=256 \
    --name="gaunt-3bpa" \
    --energy_weight=1.0 \
    --forces_weight=1000.0 \
    --max_ell=3 \
    --hidden_irreps='256x0e + 256x1o + 256x2e' \
    --r_max=5.0 \
    --num_cutoff_basis=5 \
    --correlation=3 \
    --num_radial_coupling=1 \
    --batch_size=5 \
    --num_interactions=2 \
    --weight_decay=5e-7 \
    --ema \
    --ema_decay=0.99 \
    --scaling='rms_forces_scaling' \
    --amsgrad \
    # --restart_latest \
    # --amsgrad \
