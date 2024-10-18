# leetcode-analyze
This project aims to analyze the content-topic relationship using AI model like Decision Tree.
## Target
- Extract useful content-topic relationship to help people easily disover the solution from specific content
- Use the Black-box model which is hard to explain but can give some possible topic based on content

## Tech
- Graphql
- MultiClass labeling
- KL divergence
- Oversampling
- Decision Tree

## Progress
1. Use Graphql to query all leetcode questions
2. Transform to MultiClass labeling vector
3. Oversampling until KL metric is lower
4. Training using Decision Tree
5. View the tree to find useful infomation

