## Ensure we have libssl binary
echo "deb http://security.ubuntu.com/ubuntu focal-security main" |  sudo tee /etc/apt/sources.list.d/focal-security.list
sudo apt -y update
sudo apt -y --fix-broken install
sudo apt -y install libssl1.1
sudo apt -y install xfonts-75dpi

## Install wkhtmltox binary
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_amd64.deb -O /tmp/wkhtmltox.deb
sudo dpkg -i /tmp/wkhtmltox.deb