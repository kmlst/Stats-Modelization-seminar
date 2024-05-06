# Data Valuation Using DU-Shapley
Aim of the project : How to value real world data ?  (link of the paper : https://arxiv.org/pdf/1904.02868)

## Overview
This repository contains a LaTeX document synthesizing key concepts and innovations introduced in the study of data valuation using a novel method known as Discrete Uniform Shapley (DU-Shapley). This approach aims to provide an efficient and scalable method for evaluating the worth of data in collaborative environments, particularly useful in scenarios involving large datasets.

## Contents
The document covers several core areas of data valuation:

- **Introduction**: Discusses the concept of data valuation and the necessity for innovative methods in the face of data-sharing challenges among competing entities.

- **Usual Solution & Shapley Value**: Describes traditional approaches to data valuation using the Shapley value, detailing its mathematical foundation and inherent issues such as computational complexity.

- **Contributions of DU-Shapley**: Introduces the DU-Shapley method, an efficient approximation of the Shapley value that reduces computational demands and scales better with the increase in data contributors.

- **Case Studies and Experiments**: Summarizes synthetic and real-world experiments that demonstrate the effectiveness and practical applications of DU-Shapley in various domains.

- **Discussion & Future Directions**: Provides insights into the implications of DU-Shapley for future research and its potential integration into broader data valuation frameworks.

## Main Contributions
The document highlights several important contributions of the DU-Shapley approach:

1. **Efficiency**: Significantly reduces the number of utility evaluations required, addressing the exponential growth in computations as the number of data owners increases.

2. **Scalability**: Demonstrates improved scalability and accuracy with the growing number of data contributors, making it suitable for large-scale applications.

3. **Theoretical Support**: Supported by both asymptotic and non-asymptotic theoretical guarantees, ensuring its reliability.

## Potential Applications
DU-Shapley's methodology facilitates more equitable data valuation, encouraging collaborative and incentivized data-sharing environments. It is particularly relevant in industries such as advertising and healthcare, where data sharing is crucial yet sensitive.

## Future Directions
Future enhancements could include:
- Extending the approach to accommodate heterogeneous data types and distributions.
- Integrating DU-Shapley with other valuation frameworks to explore different fairness and efficiency aspects.
- Developing accessible software implementations to integrate DU-Shapley into mainstream data analytics platforms.

## About
This project is based on the synthesis of research findings by Vianney Perchet, INRIA (Patrick Loiseau, Felipe Garrido) / CRITEO. The LaTeX document in this repository is crafted to provide a comprehensive overview of their groundbreaking work on DU-Shapley for data valuation.
