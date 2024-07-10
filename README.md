# MEA_padova
MEA analysis

## Singularity
To build the container:
```bash
singularity build spikeinterface.sif spikeinterface.def
```

To run the script:
```bash
singularity exec ./spikeinterface.sif python3 mea_padova.py
```