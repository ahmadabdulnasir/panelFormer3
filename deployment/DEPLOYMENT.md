# Deployement Steps
Start by creating a folder in the home of the user (azureuser or ubuntu)
```bash
mkdir CatVTON-Flux
```
Move into the created folder
```bash
cd CatVTON-Flux
```

## Clone the repo
```bash
git clone https://FashableMVP1@dev.azure.com/FashableMVP1/MVP/_git/CatVTON-Flux src
```

## Setup Anaconda
Install Conda if not already there
Goto anaconda website: https://www.anaconda.com/download
Download Linux version: https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
Or
Copy the link and use wget to get the file on the server
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```
Run : 
```bash Anaconda3-2024.10-1-Linux-x86_64.sh
```
Follow the prompt and accept defaults
Anaconda will be installed in: `/home/{USERNAME}}/anaconda3`
If this location `/home/{USERNAME}}/anaconda3` is Different make sure to adjust in the start script `start.virtualtryon.unet.automask.sh`

 
##Create a conda environment:
```bash
conda  create -n CatVTON-Flux python=3.10
```
### Activate the conda environment and proceed to install the project dependencies:

```bash
conda activate CatVTON-Flux
pip install -r requirements.txt 
```
Move into the src folder of the Code if not already in
```bash
cd /home/ubuntu/CatVTON-Flux/src
```
Copy and setup the start script (entry)
```bash
cp deployment/start.virtualtryon.unet.automask.sh ~/CatVTON-Flux
```
Make it executable
```bash
chmod +x ~/CatVTON-Flux/start.virtualtryon.unet.automask.sh
```
Copy the Nginx configuration:
```bash
sudo cp deployment/virtualtryon.unet.automask.nginx.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/virtualtryon.unet.automask.nginx.conf /etc/nginx/sites-enabled/
```
#### Ensure Deployment have the right IP
Check the Server IP
```bash
curl ifconfig.me
```
Copy that IP and update the Nginx conf
```bash
sudo nano /etc/nginx/sites-available/virtualtryon.unet.automask.nginx.conf
```
Find the line that says server_name and replace any IP you see (Optionally the (Sub)Domains as well) with the new Server IP
The line should look something like:
```shell
server_name virtualtryon.fashable.ai 20.4.70.193;
```

### Now test the configuration and restart nginx:
```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Systemd Service Setup
 #### Copy the service file:
```bash
sudo cp deployment/virtualtryon.unet.automask.start.uvicorn.service /etc/systemd/system/
```
#### Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable virtualtryon.unet.automask.start.uvicorn.service
sudo systemctl start virtualtryon.unet.automask.start.uvicorn.service
```

#### Check service status:
```bash
sudo systemctl status virtualtryon.unet.automask.start.uvicorn.service
```

## Updating the Application
```bash
cd ~/CatVTON-Flux/src
git pull
sudo systemctl restart virtualtryon.unet.automask.start.uvicorn.service
sudo systemctl restart nginx
```


## Troubleshooting