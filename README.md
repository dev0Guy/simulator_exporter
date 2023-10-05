# Simulate Exporter

## Overview
Simulate Exporter is a command-line tool for simulating metrics, exporting them as Prometheus metrics, and generating Kubernetes deployment files using Jinja2 templates.

## Features

- Simulate metrics from Kubernetes deployments(cpu,ram,gpu,etc..).
- Export simulated metrics in Prometheus format.
- Generate deployment files using customizable Jinja2 templates.

## Installation

```bash
pip install simulator_exporter
```
## Usage

### Simulate

Your CLI tool includes a powerful feature for simulating metrics for Kubernetes deployments. This feature is designed to monitor all deployments that include specific annotations and simulate their metrics based on configurable distributions, such as normal distributions with parameters like loc (mean) and scale (standard deviation). It leverages the scipy.stats library under the hood to perform these simulations.

#### How it Works
1. **Annotation Configuration:** To enable metric simulation, you can annotate your Kubernetes deployments with specific simulation-related annotations. For example, you might annotate a deployment with:
```yaml
annotations:
  simulate.gpu: "true"
  simulate.memory: "true"
  simulate.cpu: "true"
  simulate.distribution.type: "normal"
  simulate.distribution.arguments.loc: "100"
  simulate.distribution.arguments.scale: "20"
```
In this example, the deployment is configured to simulate GPU, memory, and CPU metrics with a normal distribution having a mean (loc) of 100 and a standard deviation (scale) of 20.

2.**Monitoring:** 
The CLI tool continuously monitors deployments with these annotations and ensures that their metrics align with the simulated distributions.

3.**Metric Simulation:**
When metrics for a deployment are requested (e.g., through a Prometheus scrape), the CLI tool generates simulated metric values based on the specified distribution parameters. For example, it generates random values that follow a normal distribution with the provided mean and standard deviation.

4.**Range Validation:**
The generated metric values are also validated to ensure they fall within appropriate ranges. Metrics should be within the specified limits, and they should not deviate significantly from the resource requests made in the deployment. This ensures that the simulated metrics are realistic and representative of the underlying resource usage.    

#### Example
Here's an example of how you can configure a deployment to simulate GPU, memory, and CPU metrics with a normal distribution:
```yaml
annotations:
  simulate.gpu: "true"
  simulate.memory: "true"
  simulate.cpu: "true"
  simulate.distribution.type: "normal"
  simulate.distribution.arguments.loc: "100"
  simulate.distribution.arguments.scale: "20"
```
This feature allows you to create realistic simulations of resource usage for your Kubernetes deployments, helping you assess how your applications would perform under various conditions.

**Genearte**
Your CLI tool allows you to generate Kubernetes deployment configurations using Jinja2 templates, providing flexibility in defining various deployment settings. You can create multiple deployment configurations with distinct parameters, resource requests, and annotations by leveraging Jinja2 templates.

#### Creating Ninja Template
Create a Jinja2 template file (e.g., templates/deployment.j2) that defines the structure of your Kubernetes deployment. In this template, you can include placeholders for values that you want to customize for each deployment.
#### Generating
To generate deployments with different settings using the Jinja2 template, follow these steps:
1. Use the generate-deployment command and specify the template file and output directory:
```bash
pytohn -m exporter generate <configfile> <template-to-use> <output-path>
```
* --template: Path to the Jinja2 template file for deployment generation.
* --output: Output directory where generated deployment files will be stored.
2. The CLI tool will create multiple deployment configuration files in the specified output directory, each with its unique values based on the template.

## Jinja2 Templates
Place your Jinja2 templates in the templates/ directory. 
These templates will be used to generate deployment files for simulation.

## Simulated Metrics
Create metric simulation scripts in the metrics/ directory. 
These scripts should generate simulated metrics data.

## Exported Metrics
The tool exposes simulated metrics in Prometheus format at the /metrics endpoint.


## Examples

### Generate Deployments
```bash 
Some info
```

### Export Metrics
Access the simulated metrics in Prometheus format:
```bash 
curl http://localhost:8080/metrics
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow the guidelines in CONTRIBUTING.md.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Mention any libraries or tools you used or were inspired by during development.

## Contact
sadmsaldkasd
asldm;asldmas;kl
mdaksdlas
