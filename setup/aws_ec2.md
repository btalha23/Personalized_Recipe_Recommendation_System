## Virtual Machine EC2 Setup via Terraform

1. Follow [Terraform Installation Guide](https://jhooq.com/install-terrafrom/) the guide to install terraform on your computer.
2. Go to [Personalized Recipe Recommender](https://github.com/btalha23/Personalized_Recipe_Recommender)
	*	Navigate to the folder <terraform/setup_ec2>
	*	Download the folder <setup_ec2> as a whole or the two files, namely
		*	main.tf
		*	variable.tf 
		to a preferred location on your computer.
3.	Navigate to the downloaded files on your computer & open `main.tf`. Fill in the `access_key` and `secret_key` of your AWS account. 
![aws_keys](../images/terraform_1.png)
If you are not sure how to get/ where to find the required keys, please refer to the document [here](setup/aws_account.md).
4.	By default `ami`is set for an Ubuntu computer. Please feel free to update this as per your preference. This can be done by updating this parameter in the `main.tf` file.
5.	By default the `instance_type` is set to `t2.micro`. However, `t2.micro` does not have enough space to download and install all the required package for running everything. It is, therefore, recommended to go to a higher tier like `t2.large` by modifying the `variable.tf` file. `CAUTION: t2.large is not available as a free tier and using this will have costs`
6.	Once you are happy with your settings in `main.tf` and `variable.tf`files, open a command prompt and navigate to the folder on your computer where you have downloaded the terraform files. ![cmd_setup_ec2](../images/terraform_2.png)
7.	