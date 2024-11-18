#!/bin/bash

# Check package manager
if command -v apt >/dev/null 2>&1; then
    PACKAGE_MANAGER="apt"
    INSTALL_COMMAND="sudo apt install -y"
elif command -v yum >/dev/null 2>&1; then
    PACKAGE_MANAGER="yum"
    INSTALL_COMMAND="sudo yum install -y"
else
    echo "[ERROR] - No compatible package manager found (apt or yum)."
    exit 1
fi

echo "[INFO] - Package manager detected: $PACKAGE_MANAGER"

# Track missing packages
MISSING_PACKAGES=()
FAILED_INSTALLS=()

# Check if Docker is available
if command -v docker >/dev/null 2>&1; then
    echo "[INFO] - Docker is installed."
else
    echo "[INFO] - Docker is NOT installed."
    echo "[INFO] - Please install Docker by following the instructions at:"
    echo "https://docs.docker.com/engine/install/"
    MISSING_PACKAGES+=("docker")
fi

# Define packages based on package manager
declare -A PACKAGES=(
    ["cmake"]="cmake"
    ["blktrace"]="blktrace"
    ["libnuma1"]="libnuma1"
    ["libnuma-dev_or_numactl-devel"]="$([[ $PACKAGE_MANAGER == 'apt' ]] && echo 'libnuma-dev' || echo 'numactl-devel')"
    ["sysstat"]="sysstat"
)

# Check each package and print installation status
for package_name in "${!PACKAGES[@]}"; do
    package="${PACKAGES[$package_name]}"
    
    if dpkg -s "$package" >/dev/null 2>&1 || rpm -q "$package" >/dev/null 2>&1; then
        echo "[INFO] - Package '$package' is already installed."
    else
        echo "[INFO] - Package '$package' is NOT installed."
        echo "[INFO] - To install, run: $INSTALL_COMMAND $package"
        MISSING_PACKAGES+=("$package")
    fi
done

# # Attempt to install missing packages, if any
# if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
#     echo "[INFO] - Attempting to install missing packages..."
#     for pkg in "${MISSING_PACKAGES[@]}"; do
#         if [[ "$pkg" == "docker" ]]; then
#             continue # Skip Docker, as we're not installing it directly through the script
#         fi
#         echo "[INFO] - Installing '$pkg'..."
#         if ! $INSTALL_COMMAND "$pkg"; then
#             echo "[ERROR] - Failed to install '$pkg'. Exiting with error code."
#             exit $?
#         fi
#     done
#     echo "[ERROR] - Some required packages were missing. Please check the above messages."
#     exit 1
# else
#     echo "[INFO] - All required packages are already installed."
# fi

# Attempt to install missing packages, if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "[INFO] - Attempting to install missing packages..."
    for pkg in "${MISSING_PACKAGES[@]}"; do
        if [[ "$pkg" == "docker" ]]; then
            continue # Skip Docker, as we're not installing it directly through the script
        fi
        echo "[INFO] - Installing '$pkg'..."
        if ! $INSTALL_COMMAND "$pkg"; then
            echo "[ERROR] - Failed to install '$pkg'."
            FAILED_INSTALLS+=("$pkg")
        fi
    done

    # Check if any installs failed
    if [ ${#FAILED_INSTALLS[@]} -gt 0 ]; then
        echo "[ERROR] - The following packages could not be installed:"
        for failed_pkg in "${FAILED_INSTALLS[@]}"; do
            echo "    - $failed_pkg"
        done
        exit 1
    else
        echo "[INFO] - All missing packages were successfully installed."
    fi
else
    echo "[INFO] - All required packages are already installed."
fi