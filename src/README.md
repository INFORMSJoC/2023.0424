# Source code for Strong Partitioning


The `Alpine` and `MPBNGCInterface` subfolders contain **modified versions** of the respective source codes that implement Strong Partitioning within Alpine. (These subfolders contain high-level details of the modifications.)

To install `Alpine` and `MPBNGCInterface`, use these modified source codes to create a development  (`dev`) version.

## Installation Instructions

To install these modified packages as development (`dev`) versions in Julia, follow these steps:

1. **Start Julia**

   Open a terminal and launch the Julia REPL:

   ```bash
   julia
   ```

2. **Activate Your Project Environment**

   At the Julia prompt, activate the environment that includes your experiment scripts (or use the default environment):
   
   ```julia
   ]
   activate .
   ```

3. **Install and Build the Modified Packages as `dev` Versions**

   Next, run the following in the Pkg mode:

   ```julia
   dev ./MPBNGCInterface
   dev ./Alpine

   build MPBNGCInterface
   build Alpine
   ```

   This installs the modified packages in development mode, so Julia uses the source code directly from the local folders.

4. **Verify Installation**

   In the Julia REPL, verify that the packages were installed correctly:

   ```julia
   using Alpine
   using MPBNGCInterface
   ```

   If no errors appear, the setup is complete.
