# How to Create a GitHub Release

Follow these steps to create a release for your Multi-Track Performance DNA Analyzer:

## Method 1: Using GitHub Web Interface (Recommended)

### Step 1: Navigate to Releases
1. Go to your repository: https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer
2. Click on "Releases" in the right sidebar (or go to the "Releases" tab)
3. Click the "Create a new release" button

### Step 2: Create a Tag
1. Click "Choose a tag"
2. Type: `v1.0.0`
3. Click "Create new tag: v1.0.0 on publish"

### Step 3: Fill in Release Details
1. **Release title**: `Multi-Track Performance DNA Analyzer v1.0.0`
2. **Description**: Copy and paste the content from `RELEASE_NOTES.md`
3. **Target**: Leave as `main` branch

### Step 4: Attach the Installer
1. Scroll down to "Attach binaries by dropping them here or selecting them"
2. Click to browse or drag and drop: `installer_output\MTP_DNA_Analyzer_Setup.exe`
3. Wait for the upload to complete (this may take a few minutes depending on file size)

### Step 5: Publish
1. Check "Set as the latest release" (should be checked by default)
2. Optionally check "Create a discussion for this release" if you want community feedback
3. Click "Publish release"

## Method 2: Using GitHub CLI (Alternative)

If you have GitHub CLI installed, you can create the release from the command line:

```bash
# Create the release
gh release create v1.0.0 ^
  --title "Multi-Track Performance DNA Analyzer v1.0.0" ^
  --notes-file RELEASE_NOTES.md ^
  "installer_output\MTP_DNA_Analyzer_Setup.exe"
```

### Installing GitHub CLI (if needed)
1. Download from: https://cli.github.com/
2. Install and authenticate: `gh auth login`
3. Follow the prompts to authenticate with your GitHub account

## After Publishing

### Update Documentation Links
After creating the release, update these files with the actual release URL:

1. **INFO_FOR_JUDGES_AND_TESTERS.md**
   - Replace `[MTP_DNA_Analyzer_Setup.exe](LINK_TO_RELEASE)` with actual link
   - Replace `[Release URL]` with actual link

2. **README.md**
   - Replace `[Releases](../../releases)` (this should work automatically)

### Share Your Release
- The release will be available at: https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases
- Direct download link will be: https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases/download/v1.0.0/MTP_DNA_Analyzer_Setup.exe

## Troubleshooting

### Upload Fails
- **File too large**: GitHub has a 2GB limit for release assets. If your installer exceeds this, consider:
  - Compressing the installer
  - Hosting on external service (Google Drive, Dropbox) and linking in release notes
  - Using GitHub Large File Storage (LFS)

### Authentication Issues
- Make sure you're logged into GitHub
- Check that you have write permissions to the repository

### Tag Already Exists
- If you need to recreate a release, delete the tag first:
  ```bash
  git tag -d v1.0.0
  git push origin :refs/tags/v1.0.0
  ```

## Next Steps

After creating the release:
1. Test the download link
2. Update documentation with the release URL
3. Announce the release (social media, forums, etc.)
4. Monitor for issues and feedback

---

**Note**: Keep the `RELEASE_NOTES.md` file for future reference and updates.
