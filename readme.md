# Deep Learning for Morphological Profiling
An end-to-end implementation of a ML System for morphological profiling using self-supervised learning to disantangle the biological signal from experimental noise in biological microscopy images.

## ML System Stack 
### Data Layer

* [Dataset RXRX1](https://www.rxrx.ai/rxrx1)
* AWS S3 for storage
* WandB for Data and model versioning 
* Data quality with [great_expectations](https://greatexpectations.io/)

### Training/Evaluation
* Pytorch and PytorchLightning 
* AWS EC2 for compute
* WandB for Experiment tracking and Hyperparameter Optimization
* Docker for resource management 

### Deployment
* FastAPI
* CircleCI
* AWS Lambda 


## Status Update
### Data
- [X] Load Data script
- [ ] Preprocess images
- [ ] Data module 

### Training
- [ ] Train resnet baseline
- [ ] Setup wandb for experiment tracking and hyperparameter optimization
- [ ] evaluation metrics
- [ ] ConvNeXt modified for 6 channels
- [ ] DenseNet 161
- [ ] Pretext task to predict rotation 

### Testing
- [ ] linting, syntax, data types.
- [ ] full training cycle
- [ ] input, outputs shapes
- [ ] single batch and epoch
- [ ] functionality test: load pretraind and predict with sample examples
- [ ] evaluation test 

### Deployment
- [ ] Setup FastAPI
- [ ] CircleCI for CI/CD
- [ ] input and output handlers
- [ ] Server script

### Monitoring & Observability 
- [ ] system health & performance
- [ ] rule-based stats(min, max, mean, std)
- [ ] data drift(mean pixel value, using projections) 
- [ ] Setup evaluation store


