# L3_chat_bot
Project 3 

### Setup Instructions with AWS CLI Installation

#### 1. **Install AWS CLI**
To simplify managing AWS credentials, installing the AWS Command Line Interface (AWS CLI) is recommended. This tool allows you to interact with AWS services using commands in your command-line shell.

**On Windows:**
- Download the AWS CLI MSI installer for Windows from the [AWS CLI official page](https://aws.amazon.com/cli/).
- Run the downloaded MSI installer and follow the on-screen instructions.

Including this screenshot so you can see how the code shows in Chat GPT, vs. how it is copied below.
 
**On Mac:**
- You can install AWS CLI on Mac using the bundled installer or Homebrew:
  - **Using Homebrew** (recommended):
    ```bash
    brew install awscli
    ```
  - **Using the bundled installer:**
    - Download the AWS CLI bundle installer using `curl`:
      ```bash
      curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
      ```
    - Install the downloaded package:
      ```bash
      sudo installer -pkg AWSCLIV2.pkg -target /
      ```

**On Linux:**
- You can install AWS CLI on Linux using `pip` or the bundled installer:
  - **Using pip**:
    ```bash
    pip install awscli --upgrade --user
    ```
  - **Using the bundled installer:**
    - Download the AWS CLI bundle installer using `curl`:
      ```bash
      curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
      ```
    - Unzip the installer and run:
      ```bash
      unzip awscliv2.zip
      sudo ./aws/install
      ```

#### 2. **Configure AWS CLI**
Once AWS CLI is installed, configure it by running the following command in the terminal. This will prompt for AWS credentials, which are necessary for using Amazon Polly.

```bash
aws configure
```
This command will ask for:
- **AWS Access Key ID** : 
- **AWS Secret Access Key** : 
- **Default region name** (e.g., `us-east-1`): us-east-1
- **Default output format** (e.g., `json`): json

Provide these details to connect the CLI with your AWS account. This is a secure method to manage AWS credentials.

#### 3. **Install Required Python Libraries**
Ensure the necessary Python libraries are installed by running the following command in your active Python environment:

```bash
pip install gradio transformers boto3 python-dotenv gtts langdetect sentencepiece sacremoses
```

#### 4. **Run the Application**
- Open the project folder with the Gradio app in VS Code.
- Open a terminal in VS Code and run the main script:
  ```bash
  python main_script.py  # Replace this with the name of your Python script
  ```

#### 5. **Using the Application**
- Access the Gradio interface via the local URL printed in the terminal. This URL can be opened in any web browser to interact with the application.

### Additional Considerations:
- **Security**: Never share your AWS credentials publicly or in shared code. Use the AWS CLI's configure command to manage credentials securely.
- **Documentation**: Ensure all scripts are well-documented to facilitate easy use and modifications by all team members.
- **Environment Variables**: If using `.env` files or other environment variable managers, ensure they are correctly configured to keep sensitive information secure.

By following these steps, each team member should be able to install AWS CLI, configure it, and run the Gradio app effectively within their development environment.
