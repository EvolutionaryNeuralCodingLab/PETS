import numpy as np
import h5py
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Optional, Dict, List


class OERecording:
    """
    Class for accessing Open Ephys format recordings.
    
    This class can work in two modes:
    1. Legacy mode: Loads metadata from MATLAB-generated .mat files (backward compatible)
    2. Standalone mode: Extracts metadata directly from Open Ephys recording files
    
    Usage:
        # Legacy mode (backward compatible)
        rec = OERecording(Path("path/to/OE_metaData.mat"))
        
        # Standalone mode (new)
        rec = OERecording(Path("path/to/recording/directory"))
    """

    def group_to_dict(self, group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = self.group_to_dict(item)
            else:

                result[key] = item[()]  # Convert dataset to NumPy array and assign its value
        return result

    # a helper function to sort recording files by their correct numbering scheme
    @staticmethod
    def extract_number_from_file(filename, suffix):
        match = re.search(r'(\d+)\.' + suffix + '$', filename)
        if match:
            return int(match.group(1))
        else:
            return None

    @staticmethod
    def _parse_continuous_header(file_path: Path) -> Dict:
        """
        Parse the 1024-byte header from an Open Ephys .continuous file.
        
        The header is a text file with MATLAB-style variable assignments.
        Returns a dictionary with header fields.
        """
        header = {}
        with open(file_path, 'rb') as f:
            hdr_bytes = f.read(1024)
            hdr_text = hdr_bytes.decode('utf-8', errors='ignore')
        
        # Parse header fields (MATLAB-style assignments)
        # Pattern: header.field = value;
        pattern = r'header\.(\w+)\s*=\s*([^;]+);'
        matches = re.findall(pattern, hdr_text)
        
        for field, value in matches:
            value = value.strip().strip("'\"")  # Remove quotes
            # Try to convert to appropriate type
            try:
                # Try float first
                if '.' in value:
                    header[field] = float(value)
                else:
                    header[field] = int(value)
            except ValueError:
                # Keep as string
                header[field] = value
        
        # Ensure version exists (default to 0.0 if not found)
        if 'version' not in header:
            header['version'] = 0.0
        
        return header

    @staticmethod
    def _parse_xml_metadata(recording_dir: Path) -> Optional[Dict]:
        """
        Parse Open Ephys XML files (settings.xml and structure.openephys).
        
        Returns a dictionary with channel information, or None if XML files don't exist.
        """
        xml_data = {}
        settings_path = recording_dir / 'settings.xml'
        structure_path = recording_dir / 'structure.openephys'
        
        if not settings_path.exists():
            return None
        
        try:
            settings_tree = ET.parse(settings_path)
            settings_root = settings_tree.getroot()
            
            # Get version
            version_elem = settings_root.find('.//INFO/VERSION')
            version_str = version_elem.text if version_elem is not None else "0.0"
            xml_data['version'] = version_str
            
            # Parse version number (handle versions like "0.6.7")
            try:
                version_float = float(version_str.split('.')[0] + '.' + version_str.split('.')[1] if '.' in version_str else version_str)
            except:
                version_float = 0.0
            
            # Check if version > 0.6.0 (uses structure.openephys)
            if structure_path.exists() and version_float > 0.6:
                structure_tree = ET.parse(structure_path)
                structure_root = structure_tree.getroot()
                
                # Get recording stream
                recording = structure_root.find('RECORDING')
                if recording is not None:
                    stream = recording.find('STREAM')
                    if stream is not None:
                        # Get channels
                        channels = stream.findall('CHANNEL')
                        xml_data['channelNamesAll'] = [ch.get('name') for ch in channels]
                        xml_data['channelFiles'] = [ch.get('filename') for ch in channels]
                        
                        # Extract channel type names (last part after underscore)
                        xml_data['chTypeName'] = [name.split('_')[-1] if '_' in name else name 
                                                 for name in xml_data['channelNamesAll']]
                        
                        # Extract channel numbers
                        xml_data['channelNumbers'] = []
                        for ch_type in xml_data['chTypeName']:
                            nums = re.findall(r'\d+', ch_type)
                            xml_data['channelNumbers'].append(int(nums[0]) if nums else 0)
                        
                        # Classify channels
                        xml_data['pAnalogCh'] = [ch[0] in ['A', 'P'] for ch in xml_data['chTypeName']]
                        xml_data['pCh'] = [ch[0] == 'C' for ch in xml_data['chTypeName']]
                        
                        # Get event file name
                        events = stream.find('EVENTS')
                        if events is not None:
                            xml_data['eventFileName'] = events.get('filename', 'all_channels.events')
            else:
                # Version <= 0.6.0: use settings.xml
                # Find Rhythm FPGA processor
                processors = settings_root.findall('.//PROCESSOR')
                rhythm_processors = [p for p in processors 
                                   if p.get('pluginName') == 'Rhythm FPGA']
                
                if rhythm_processors:
                    proc = rhythm_processors[0]
                    # Get channel info
                    channel_info = proc.find('CHANNEL_INFO')
                    if channel_info is not None:
                        channels = channel_info.findall('CHANNEL')
                        xml_data['chTypeName'] = [ch.get('name') for ch in channels]
                        
                        # Get selection state
                        selections = proc.findall('.//SELECTIONSTATE')
                        xml_data['isRecorded'] = [s.get('param') == '1' for s in selections]
                        
                        # Filter to recorded channels
                        xml_data['channelNames'] = [name for name, rec in 
                                                   zip(xml_data['chTypeName'], xml_data['isRecorded']) 
                                                   if rec]
                        
                        # Extract channel numbers
                        xml_data['channelNumbers'] = []
                        for ch_type in xml_data['channelNames']:
                            nums = re.findall(r'\d+', ch_type)
                            xml_data['channelNumbers'].append(int(nums[0]) if nums else 0)
                        
                        # Classify channels
                        xml_data['pAnalogCh'] = [ch[0] == 'A' for ch in xml_data['channelNames']]
                        xml_data['pCh'] = [ch[0] == 'C' for ch in xml_data['channelNames']]
            
            xml_data['settings_root'] = settings_root
            if structure_path.exists():
                structure_tree = ET.parse(structure_path)
                xml_data['structure_root'] = structure_tree.getroot()
            else:
                xml_data['structure_root'] = None
            
            return xml_data
        except Exception as e:
            print(f'Warning: Error parsing XML files: {e}')
            return None

    def _build_blk_cont(self, software_version: float) -> Dict:
        """
        Build the blkCont structure for continuous data records.
        
        This structure defines the binary format of each record in .continuous files.
        """
        b_str = ['ts', 'nsamples', 'recNum', 'data', 'recordMarker']
        b_types = ['int64', 'uint16', 'uint16', 'int16', 'uint8']
        b_repeat = [1, 1, 1, self.dataSamplesPerRecord, 10]
        
        # Version-specific adjustments
        if software_version < 0.2:
            # Remove recNum field
            b_str = b_str[:2] + b_str[3:]
            b_types = b_types[:2] + b_types[3:]
            b_repeat = b_repeat[:2] + b_repeat[3:]
        
        if software_version < 0.1:
            # Use uint64 for timestamps, int16 for sample count
            b_types[0] = 'uint64'
            b_types[1] = 'int16'
        
        blk_cont = {
            'Str': b_str,
            'Types': b_types,
            'Repeat': b_repeat
        }
        
        # Calculate bytes per field
        type_sizes = {
            'int64': 8, 'uint64': 8, 'int32': 4, 'uint32': 4,
            'int16': 2, 'uint16': 2, 'int8': 1, 'uint8': 1
        }
        
        blk_bytes = []
        for dtype, repeat in zip(b_types, b_repeat):
            size = type_sizes.get(dtype, 1)
            blk_bytes.append(size * repeat)
        
        blk_cont['Bytes'] = blk_bytes
        blk_cont['bytesPerRec'] = sum(blk_bytes)
        
        return blk_cont

    def _build_blk_evnt(self, software_version: float) -> Dict:
        """
        Build the blkEvnt structure for event data records.
        """
        b_str = ['timestamps', 'sampleNum', 'eventType', 'nodeId', 'eventId', 'data', 'recNum']
        b_types = ['int64', 'uint16', 'uint8', 'uint8', 'uint8', 'uint8', 'uint16']
        b_repeat = [1, 1, 1, 1, 1, 1, 1]
        
        # Version-specific adjustments
        if software_version < 0.2:
            # Remove recNum field
            b_str = b_str[:-1]
            b_types = b_types[:-1]
            b_repeat = b_repeat[:-1]
        
        if software_version < 0.1:
            # Use uint64 for timestamps
            b_types[0] = 'uint64'
        
        blk_evnt = {
            'Str': b_str,
            'Types': b_types,
            'Repeat': b_repeat
        }
        
        # Calculate bytes per field
        type_sizes = {
            'int64': 8, 'uint64': 8, 'int32': 4, 'uint32': 4,
            'int16': 2, 'uint16': 2, 'int8': 1, 'uint8': 1
        }
        
        blk_bytes = []
        for dtype, repeat in zip(b_types, b_repeat):
            size = type_sizes.get(dtype, 1)
            blk_bytes.append(size * repeat)
        
        blk_evnt['Bytes'] = blk_bytes
        blk_evnt['bytesPerRec'] = sum(blk_bytes)
        
        return blk_evnt

    def read_events_to_dataframe(self, align_to_zero: bool = True,
                                  zeroth_sample: Optional[int] = None) -> 'pd.DataFrame':
        """
        Read TTL events from the binary .events file and return a DataFrame compatible with
        BlockSync's oe_events_parser (columns: line, state, sample_number).
        
        Works for both OE 0.5.x (legacy) and 0.6.x formats. Uses version from .continuous
        header to determine binary record structure (see OERecording.m).
        
        Parameters
        ----------
        align_to_zero : bool
            If True, subtract zeroth_sample from sample_number (align to acquisition start).
        zeroth_sample : int, optional
            First acquired sample number. If None, uses globalStartTime_ms * sample_rate / 1000.
            
        Returns
        -------
        pd.DataFrame
            Columns: line (TTL channel 0-8), state (1=rising, 0=falling), sample_number.
        """
        import pandas as pd
        
        event_file_name = getattr(self, 'eventFileName', 'all_channels.events')
        event_file = self.oe_file_path / event_file_name
        if not event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")
        
        n_records = getattr(self, 'nRecordsEvnt', 0)
        if n_records == 0:
            # Infer from file size if not set (e.g. legacy mat load)
            evnt_file_size = getattr(self, 'evntFileSize', None)
            if evnt_file_size and hasattr(self, 'bytesPerRecEvnt'):
                n_records = int((evnt_file_size - self.headerSizeByte) / self.bytesPerRecEvnt)
            if n_records == 0:
                return pd.DataFrame(columns=["line", "state", "sample_number"])
        
        # Build record structure for reading (blkEvnt from _build_blk_evnt)
        software_version = getattr(self, 'softwareVersion', None)
        if software_version is None or (hasattr(software_version, '__len__') and len(software_version) > 0):
            try:
                v = self.softwareVersion
                software_version = float(v[0]) if hasattr(v, '__len__') and len(v) > 0 else float(v)
            except (TypeError, IndexError):
                software_version = 0.6  # assume newer if unclear
        else:
            software_version = float(software_version)
        
        blk_evnt = self._build_blk_evnt(software_version)
        bytes_per_rec = blk_evnt['bytesPerRec']
        blk_bytes = blk_evnt['Bytes']
        blk_str = blk_evnt['Str']
        blk_types = blk_evnt['Types']
        
        # Field offsets (bytes from start of record, after 1024 header)
        type_sizes = {'int64': 8, 'uint64': 8, 'uint16': 2, 'uint8': 1}
        
        def read_field(fid, field_name, n_records):
            idx = blk_str.index(field_name)
            dtype_str = blk_types[idx]
            dtype = np.dtype('<i8' if dtype_str == 'int64' else 
                           '<u8' if dtype_str == 'uint64' else
                           '<u2' if dtype_str == 'uint16' else '<u1')
            field_bytes = blk_bytes[idx]
            offset = self.headerSizeByte + sum(blk_bytes[:idx])
            skip = bytes_per_rec - field_bytes
            fid.seek(offset)
            arr = np.zeros(n_records, dtype=dtype)
            for i in range(n_records):
                arr[i] = np.frombuffer(fid.read(field_bytes), dtype=dtype)[0]
                if i < n_records - 1:
                    fid.seek(skip, 1)
            return arr
        
        with open(event_file, 'rb') as f:
            timestamps = read_field(f, 'timestamps', n_records)
            event_type = read_field(f, 'eventType', n_records)
            event_id = read_field(f, 'eventId', n_records)
            data_ch = read_field(f, 'data', n_records)
        
        # TTL events only: eventType == 3
        p_ttl = (event_type == 3)
        timestamps = timestamps[p_ttl].astype(np.int64)
        event_id = event_id[p_ttl]
        data_ch = data_ch[p_ttl]
        
        # state: 1 = rising (eventId==1), 0 = falling (eventId==0)
        state = (event_id == 1).astype(np.int32)
        
        if align_to_zero:
            if zeroth_sample is None:
                zeroth_sample = int(self.globalStartTime_ms * (self.samplingFrequency[0] / 1000))
            sample_number = timestamps - zeroth_sample
        else:
            sample_number = timestamps
        
        df = pd.DataFrame({
            "line": data_ch.astype(np.int32),
            "state": state,
            "sample_number": sample_number.astype(np.int64)
        })
        return df

    def _extract_timestamps(self, file_path: Path, blk_cont: Dict, 
                           n_records: int, sampling_frequency: float) -> np.ndarray:
        """
        Extract timestamps from the first channel file.
        
        This replicates MATLAB's fread pattern:
        fread(fid, nRecordsCont*blkCont(1).Repeat, sprintf('%d*%s', ...), 
              bytesPerRecCont - blkBytesCont(1), 'l')
        
        Returns timestamps in milliseconds (not yet relative to start).
        """
        with open(file_path, 'rb') as f:
            # Skip header
            f.seek(self.headerSizeByte)
            
            # Determine timestamp type based on version
            timestamp_type = blk_cont['Types'][0]  # First field is timestamp
            timestamp_repeat = blk_cont['Repeat'][0]  # Usually 1
            
            if timestamp_type == 'uint64':
                dtype = np.dtype('<u8')  # little-endian uint64
            else:
                dtype = np.dtype('<i8')  # little-endian int64
            
            # Get bytes per record and timestamp bytes
            bytes_per_rec = blk_cont.get('bytesPerRec', sum(blk_cont.get('Bytes', [])))
            blk_bytes = blk_cont.get('Bytes', [])
            timestamp_bytes = blk_bytes[0] if len(blk_bytes) > 0 else 8
            # MATLAB: skip bytesPerRecCont - blkBytesCont(1)
            skip_bytes = bytes_per_rec - timestamp_bytes
            
            # Read all timestamps at once using numpy's fromfile with proper skipping
            # MATLAB: fread(fid, nRecordsCont*Repeat, type, skip_bytes, 'l')
            # We need to read nRecordsCont * Repeat values, skipping skip_bytes between each
            
            timestamps = []
            for i in range(n_records):
                # Read timestamp
                ts_bytes = f.read(timestamp_bytes)
                if len(ts_bytes) < timestamp_bytes:
                    break  # End of file
                ts = np.frombuffer(ts_bytes, dtype=dtype)[0]
                timestamps.append(ts)
                # Skip to next record
                if i < n_records - 1:  # Don't skip after last record
                    f.seek(skip_bytes, 1)
        
        timestamps = np.array(timestamps, dtype=dtype)
        # Convert to milliseconds (MATLAB: /samplingFrequency * 1000)
        timestamps_ms = timestamps.astype(np.float64) / sampling_frequency * 1000
        
        return timestamps_ms

    def extract_metadata(self, recording_dir: Path):
        """
        Extract metadata from Open Ephys recording files.
        
        This method replicates the functionality of MATLAB's extractMetaData().
        """
        print(f'\nExtracting meta data from: {recording_dir}...')
        
        # Find all .continuous files
        continuous_files = sorted([f for f in recording_dir.glob(f'*.{self.fileExtension}')])
        n_files = len(continuous_files)
        
        if n_files == 0:
            raise ValueError(f'No .{self.fileExtension} files found in {recording_dir}')
        
        # Initialize file data structure
        file_data = {
            'fileSize': [],
            'samplingFrequency': [],
            'MicrovoltsPerAD': [],
            'startDate': [],
            'blockLength': [],
            'dataDescriptionCont': [],
            'fileHeaders': [],
            'softwareVersion': [],
            'channelName': [],
            'channelType': []
        }
        
        # Extract metadata from each file
        for file_path in continuous_files:
            # Get file size
            file_size = file_path.stat().st_size
            file_data['fileSize'].append(file_size)
            
            # Parse header
            header = self._parse_continuous_header(file_path)
            file_data['fileHeaders'].append(header)
            
            # Extract fields
            file_data['samplingFrequency'].append(header.get('sampleRate', 0))
            file_data['MicrovoltsPerAD'].append(header.get('bitVolts', 0))
            file_data['startDate'].append(header.get('date_created', ''))
            file_data['blockLength'].append(header.get('blockLength', 1024))
            file_data['dataDescriptionCont'].append(header.get('description', ''))
            file_data['channelType'].append(header.get('channelType', ''))
            file_data['softwareVersion'].append(header.get('version', 0.0))
            file_data['channelName'].append(header.get('channel', ''))
        
        # Parse XML files
        xml_data = self._parse_xml_metadata(recording_dir)
        
        if xml_data:
            self.openEphyXMLData = xml_data.get('settings_root')
            if 'structure_root' in xml_data:
                self.openEphyXMLStructureData = xml_data['structure_root']
            else:
                self.openEphyXMLStructureData = None
            
            # Verify consistency between XML and file data
            if 'channelNames' in xml_data:
                xml_channel_names = xml_data['channelNames']
                file_channel_names = file_data['channelName']
                if len(xml_channel_names) == len(file_channel_names):
                    common = set(xml_channel_names) & set(file_channel_names)
                    if len(common) == len(file_channel_names):
                        print('xml data matches file data.')
                    else:
                        print('Warning: There is mismatch between the settings xml and the data files!')
        else:
            print('settings.xml file not found! Extracting info from continuous recording files.')
            self.openEphyXMLData = None
            self.openEphyXMLStructureData = None
        
        # Extract channel types and numbers from file names
        file_data['type'] = []
        file_data['channelNumbers'] = []
        for ch_name in file_data['channelName']:
            # Extract type (e.g., 'CH', 'ADC', 'AUX')
            type_match = re.findall(r'\S+', ch_name)
            file_data['type'].append(type_match[0] if type_match else '')
            
            # Extract channel number
            num_match = re.findall(r'\d+$', ch_name)
            file_data['channelNumbers'].append(int(num_match[0]) if num_match else 0)
        
        # Classify channels (digital vs analog)
        # Determine source node type from XML if available
        if xml_data and xml_data.get('version', '0.0') != '0.0':
            version_str = xml_data['version']
            try:
                # Parse version number (handle versions like "0.6.7")
                version = float(version_str.split('.')[0] + '.' + version_str.split('.')[1] if '.' in version_str else version_str)
            except:
                version = 0.0
            if version > 0.6 and self.openEphyXMLStructureData is not None:
                # Check source node name
                recording = self.openEphyXMLStructureData.find('RECORDING')
                if recording is not None:
                    stream = recording.find('STREAM')
                    source_node = stream.get('source_node_name') if stream is not None else None
                    if source_node == 'XDAQ':
                        p_analog_ch = [i for i, t in enumerate(file_data['type']) 
                                     if t and (t[0] in ['A', 'P'])]
                        p_ch = [i for i, t in enumerate(file_data['type']) 
                               if t and t[0] == 'C']
                    else:
                        p_analog_ch = [i for i, t in enumerate(file_data['type']) 
                                     if t and (t[:3] in ['ADC', 'C1_'])]
                        p_ch = [i for i, t in enumerate(file_data['type']) 
                               if t and t[:2] == 'CH']
                else:
                    p_analog_ch = [i for i, t in enumerate(file_data['type']) 
                                 if t and (t[:3] in ['ADC', 'C1_'])]
                    p_ch = [i for i, t in enumerate(file_data['type']) 
                           if t and t[:2] == 'CH']
            else:
                p_analog_ch = [i for i, t in enumerate(file_data['type']) 
                             if t and (t[:3] in ['ADC', 'AUX', 'C1_'])]
                p_ch = [i for i, t in enumerate(file_data['type']) 
                       if t and t[:2] == 'CH']
        else:
            p_analog_ch = [i for i, t in enumerate(file_data['type']) 
                         if t and (t[:3] in ['ADC', 'AUX', 'C1_'])]
            p_ch = [i for i, t in enumerate(file_data['type']) 
                   if t and t[:2] == 'CH']
        
        # Sort channels by number
        if p_analog_ch:
            analog_nums = [file_data['channelNumbers'][i] for i in p_analog_ch]
            p_analog_ch = [p_analog_ch[i] for i in np.argsort(analog_nums)]
        
        if p_ch:
            ch_nums = [file_data['channelNumbers'][i] for i in p_ch]
            p_ch = [p_ch[i] for i in np.argsort(ch_nums)]
        
        # Organize channel files
        self.channelFilesAnalog = [continuous_files[i].name for i in p_analog_ch]
        self.channelFiles = [continuous_files[i].name for i in p_ch]
        
        # Get channel numbers
        if len(p_ch) > 0:
            self.channelNumbers = np.array([file_data['channelNumbers'][i] for i in p_ch])
            if np.any(self.channelNumbers == 0):
                self.channelNumbers = self.channelNumbers + 1
        else:
            self.channelNumbers = np.array([])
        
        if len(p_analog_ch) > 0:
            self.analogChannelNumbers = np.array([file_data['channelNumbers'][i] for i in p_analog_ch])
            if np.any(self.analogChannelNumbers == 0):
                self.analogChannelNumbers = self.analogChannelNumbers + 1
        else:
            self.analogChannelNumbers = np.array([])
        
        # Sort channel numbers
        if len(self.channelNumbers) > 0:
            sort_idx = np.argsort(self.channelNumbers)
            self.channelNumbers = self.channelNumbers[sort_idx]
            self.channelFiles = [self.channelFiles[i] for i in sort_idx]
            self.channelNames = [file_data['channelName'][p_ch[i]] for i in sort_idx]
            # Create n2s mapping
            self.n2s = {}
            for idx, ch_num in enumerate(self.channelNumbers):
                self.n2s[ch_num] = idx + 1
        else:
            self.channelNames = []
            self.n2s = {}
        
        # Handle analog channels
        n_analog = len(self.analogChannelNumbers)
        if n_analog > 0:
            if len(np.unique(self.analogChannelNumbers)) != n_analog:
                # Duplicate numbers - reorder serially
                print('Analog channel numbers contain duplicates!!! Reordering numbers serially.')
                self.analogChannelNumbers = np.arange(1, n_analog + 1)
                sort_idx = np.arange(n_analog)
            else:
                sort_idx = np.argsort(self.analogChannelNumbers)
                self.analogChannelNumbers = self.analogChannelNumbers[sort_idx]
            
            self.channelFilesAnalog = [self.channelFilesAnalog[i] for i in sort_idx]
            self.analogChannelNames = [file_data['channelName'][p_analog_ch[i]] for i in sort_idx]
            # Create n2sA mapping
            self.n2sA = {}
            for idx, ch_num in enumerate(self.analogChannelNumbers):
                self.n2sA[ch_num] = idx + 1
        else:
            self.analogChannelNames = []
            self.n2sA = {}
        
        # Set properties from file data
        if len(p_ch) > 0:
            self.MicrovoltsPerAD = np.array([file_data['MicrovoltsPerAD'][i] for i in p_ch])
            self.samplingFrequency = np.array([file_data['samplingFrequency'][i] for i in p_ch])
            self.fileSize = np.array([file_data['fileSize'][i] for i in p_ch])
            self.softwareVersion = np.array([file_data['softwareVersion'][i] for i in p_ch])
            self.blockLength = np.array([file_data['blockLength'][i] for i in p_ch])
            self.startDate = [file_data['startDate'][i] for i in p_ch]
            self.fileHeaders = [file_data['fileHeaders'][i] for i in p_ch]
            self.dataDescriptionCont = [file_data['dataDescriptionCont'][i] for i in p_ch]
        else:
            self.MicrovoltsPerAD = np.array([])
            self.samplingFrequency = np.array([])
            self.fileSize = np.array([])
            self.softwareVersion = np.array([])
            self.blockLength = np.array([])
            self.startDate = []
            self.fileHeaders = []
            self.dataDescriptionCont = []
        
        if len(p_analog_ch) > 0:
            self.MicrovoltsPerADAnalog = np.array([file_data['MicrovoltsPerAD'][i] for i in p_analog_ch]) * 1e6
            self.samplingFrequencyAnalog = np.array([file_data['samplingFrequency'][i] for i in p_analog_ch])
            self.fileSizeAnalog = np.array([file_data['fileSize'][i] for i in p_analog_ch])
            self.softwareVersionAnalog = np.array([file_data['softwareVersion'][i] for i in p_analog_ch])
            self.blockLengthAnalog = np.array([file_data['blockLength'][i] for i in p_analog_ch])
            self.startDateA = [file_data['startDate'][i] for i in p_analog_ch]
            self.fileHeadersAnalog = [file_data['fileHeaders'][i] for i in p_analog_ch]
            self.dataDescriptionContAnalog = [file_data['dataDescriptionCont'][i] for i in p_analog_ch]
        else:
            self.MicrovoltsPerADAnalog = np.array([])
            self.samplingFrequencyAnalog = np.array([])
            self.fileSizeAnalog = np.array([])
            self.softwareVersionAnalog = np.array([])
            self.blockLengthAnalog = np.array([])
            self.startDateA = []
            self.fileHeadersAnalog = []
            self.dataDescriptionContAnalog = []
        
        self.ZeroADValue = np.zeros_like(self.MicrovoltsPerAD)
        self.ZeroADValueAnalog = np.zeros_like(self.MicrovoltsPerADAnalog)
        
        # Build metadata structures
        software_version = file_data['softwareVersion'][0] if file_data['softwareVersion'] else 0.0
        blk_cont_dict = self._build_blk_cont(software_version)
        # Convert to format expected by existing code
        self.blkCont = {
            'Repeat': blk_cont_dict['Repeat'],
            'Types': blk_cont_dict['Types'],
            'Str': blk_cont_dict['Str'],
            'Bytes': blk_cont_dict['Bytes'],  # Keep Bytes for timestamp extraction
            'bytesPerRec': blk_cont_dict['bytesPerRec']
        }
        self.blkBytesCont = np.array(blk_cont_dict['Bytes'])
        self.bytesPerRecCont = blk_cont_dict['bytesPerRec']
        
        # Extract timestamps
        print('\nExtracting time stamp information...')
        if len(self.channelNumbers) == 0:
            print('0 electrode channels found in recording!')
            if len(self.analogChannelNumbers) > 0:
                self.samplingFrequency = np.array([self.samplingFrequencyAnalog[0]])
                fid_for_timestamps = recording_dir / self.channelFilesAnalog[0]
                self.nRecordsCont = int(np.floor((self.fileSizeAnalog[0] - self.headerSizeByte) / self.bytesPerRecCont))
            else:
                raise ValueError('No channels found in recording!')
        else:
            fid_for_timestamps = recording_dir / self.channelFiles[0]
            self.nRecordsCont = int(np.floor((self.fileSize[0] - self.headerSizeByte) / self.bytesPerRecCont))
        
        self.recordLength = self.dataSamplesPerRecord / self.samplingFrequency[0] * 1000
        self.sample_ms = 1000 / self.samplingFrequency[0]
        
        # Extract timestamps (MATLAB: fread pattern)
        self.allTimeStamps = self._extract_timestamps(
            fid_for_timestamps, self.blkCont, self.nRecordsCont, self.samplingFrequency[0]
        )
        
        # MATLAB: obj.globalStartTime_ms=obj.allTimeStamps(1);
        self.globalStartTime_ms = self.allTimeStamps[0]
        
        # MATLAB: obj.allTimeStamps = obj.allTimeStamps-obj.globalStartTime_ms;
        self.allTimeStamps = self.allTimeStamps - self.globalStartTime_ms
        
        # MATLAB: if obj.allTimeStamps(end)<=obj.allTimeStamps(end-1)
        if len(self.allTimeStamps) > 1 and self.allTimeStamps[-1] <= self.allTimeStamps[-2]:
            print('Last record has a non-valid timestamp!!! Removing last data record.')
            self.allTimeStamps = self.allTimeStamps[:-1]
            self.nRecordsCont -= 1
        
        # MATLAB: obj.recordingDuration_ms=obj.allTimeStamps(end);
        self.recordingDuration_ms = self.allTimeStamps[-1] if len(self.allTimeStamps) > 0 else 0.0
        
        # Ensure allTimeStamps is in the format expected by get_data (2D array)
        # The existing code expects allTimeStamps[0] to be the array
        if self.allTimeStamps.ndim == 1:
            self.allTimeStamps = self.allTimeStamps.reshape(1, -1)
        
        # Check for missing blocks
        ts_diff = np.diff(self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps)
        if np.any(ts_diff > self.dataSamplesPerRecord / self.samplingFrequency[0] * 1000):
            print('\nError!!! Some blocks are missing in recording!!!')
        
        # Check data integrity
        print('\nChecking integrity of all records in ch1...')
        with open(fid_for_timestamps, 'rb') as f:
            f.seek(self.headerSizeByte + self.blkBytesCont[0])  # Skip header and timestamp
            # Read sample numbers with proper skipping
            sample_numbers = []
            bytes_per_rec = self.bytesPerRecCont
            sample_num_offset = self.blkBytesCont[0]  # After timestamp
            for i in range(min(self.nRecordsCont, 1000)):  # Check first 1000 records only
                f.seek(self.headerSizeByte + i * bytes_per_rec + sample_num_offset)
                sample_num = np.frombuffer(f.read(2), dtype='<u2')[0]  # uint16, little-endian
                sample_numbers.append(sample_num)
            
            sample_numbers = np.array(sample_numbers)
            # Allow some tolerance - last record might be shorter
            valid_samples = np.sum(sample_numbers == self.dataSamplesPerRecord)
            if valid_samples < len(sample_numbers) * 0.99 and software_version >= 0.1:
                print(f'Warning: {len(sample_numbers) - valid_samples} records have unexpected sample counts')
                # Don't raise error, just warn
        
        # Build event structure
        self.blkEvnt = self._build_blk_evnt(software_version)
        self.blkBytesEvnt = np.array(self.blkEvnt['Bytes'])
        self.bytesPerRecEvnt = self.blkEvnt['bytesPerRec']
        
        # Get event file
        if xml_data and 'eventFileName' in xml_data:
            self.eventFileName = xml_data['eventFileName']
        else:
            self.eventFileName = 'all_channels.events'
        
        event_file = recording_dir / self.eventFileName
        if event_file.exists():
            self.evntFileSize = event_file.stat().st_size
            self.nRecordsEvnt = int(np.floor((self.evntFileSize - self.headerSizeByte) / self.bytesPerRecEvnt))
        else:
            self.evntFileSize = 0
            self.nRecordsEvnt = 0
        
        print('\nMetadata extraction complete.')
        
        # Ensure recordingDuration_ms is a scalar (not array)
        if isinstance(self.recordingDuration_ms, np.ndarray):
            self.recordingDuration_ms = self.recordingDuration_ms.item() if self.recordingDuration_ms.size == 1 else self.recordingDuration_ms[0]

    def __init__(self, oe_path: Union[Path, str]):
        """
        Initialize OERecording object.
        
        Args:
            oe_path: Either a path to a MATLAB metadata .mat file (legacy mode)
                    or a path to the recording directory (standalone mode)
        """
        # Convert to Path if string
        oe_path = Path(oe_path)
        
        # Initialize constants
        self.headerSizeByte = 1024
        self.fileExtension = 'continuous'
        self.eventFileExtension = 'events'
        self.signalBits = 16  # the quantization of the sampling card
        self.dataSamplesPerRecord = 1024
        self.maxTTLBit = 9
        
        # Determine mode: check if path is a .mat file or contains 'metaData'
        is_legacy_mode = (oe_path.suffix == '.mat' or 
                         'metaData' in oe_path.name or
                         (oe_path.is_file() and oe_path.suffix == '.mat'))
        
        if is_legacy_mode:
            # Legacy mode: load from MATLAB metadata file
            self._load_from_mat_file(oe_path)
        else:
            # Standalone mode: extract metadata from recording files
            if not oe_path.is_dir():
                raise ValueError(f'Path {oe_path} is not a directory. '
                               f'For legacy mode, provide a .mat metadata file.')
            self.oe_file_path = oe_path
            self.extract_metadata(oe_path)
            
            # Set up file lists for compatibility
            self.channel_files = self.channelFiles
            self.analog_files = self.channelFilesAnalog
            self.accel_files = sorted([f.name for f in oe_path.glob('*AUX*.continuous')],
                                      key=lambda x: self.extract_number_from_file(x, suffix='continuous'))
    
    def _load_from_mat_file(self, mat_file_path: Path):
        """
        Load metadata from MATLAB-generated .mat file (legacy mode).
        """
        # initialize some critical variables
        self.allTimeStamps = None
        self.sample_ms = None
        self.channelNumbers = None
        self.bytesPerRecCont = None
        self.recordLength = None
        self.blkCont = None
        self.blkBytesCont = None
        self.MicrovoltsPerAD = None
        self.mat_file = None
        self.globalStartTime_ms = None

        # create the metadata_dict object:
        # open the mat file:
        try:
            self.mat_file = h5py.File(str(mat_file_path), 'r')
        except Exception as e:
            print(f'An error occurred while trying to reach {str(mat_file_path)}, check matlab output format!')
            print(f'Error: {e}')
            if self.mat_file is not None:
                self.mat_file.close()
            raise

        # implement on the metaData group:
        meta_dict = self.group_to_dict(self.mat_file['metaData'])

        # resolve internal references of blkCont object:
        # get to the blkCont mat and reform it from reference instances
        blk_cont_dict = {
            'Repeat': [],
            'Types': [],
            'Str': []
        }
        blk_cont_group = self.mat_file['metaData/blkCont']
        for i in blk_cont_group['Repeat']:
            res = np.array(self.mat_file[i[0]][0])
            blk_cont_dict['Repeat'].append(res[0])

        for i in blk_cont_group['Types']:
            res = np.array((self.mat_file[i[0]]))
            str_array = np.vectorize(chr)(res).flatten()
            str_value = ''.join(str_array.flatten())
            blk_cont_dict['Types'].append(str_value)

        for i in blk_cont_group['Str']:
            res = np.array(self.mat_file[i[0]])
            str_array = np.vectorize(chr)(res).flatten()
            str_value = ''.join(str_array.flatten())
            blk_cont_dict['Str'].append(str_value)

        # close the file
        self.mat_file.close()

        # switch out the dictionary blkCont attribute
        meta_dict['blkCont'] = blk_cont_dict
        # parse a dictionary into attributes of the class
        for key, value in meta_dict.items():
            setattr(self, key, value)

        # Set file path
        self.oe_file_path = mat_file_path.parent

        # get the channel files but also sort them by their true numbering scheme
        self.channel_files = sorted([i.name for i in mat_file_path.parent.iterdir() if
                                     ('.continuous' in str(i)) & ('AUX' not in str(i)) & ('ADC' not in str(i))],
                                    key=lambda x: self.extract_number_from_file(x, suffix='continuous'))

        self.analog_files = sorted([i.name for i in mat_file_path.parent.iterdir() if 'ADC' in str(i)],
                                   key=lambda x: self.extract_number_from_file(x, suffix='continuous'))
        self.accel_files = sorted([i.name for i in mat_file_path.parent.iterdir() if ('AUX' in str(i))],
                                  key=lambda x: self.extract_number_from_file(x, suffix='continuous'))

    def get_data(self, channels,
                 start_time_ms,
                 window_ms,
                 convert_microvolts=True,
                 return_timestamps=True,
                 repress_output=False):
        """
        Retrieve continuous data from Open Ephys .continuous files (neural/headstage channels).

        Open Ephys convention: For headstage channels, the .continuous header field `bitVolts`
        is in **microvolts per AD count**. So: voltage_µV = raw_int16 * bitVolts.

        :param channels: channel numbers to sample from [1 x N]
        :param start_time_ms: window start times [1 x N] in ms
        :param window_ms: length of each sampling window in ms
        :param convert_microvolts: when True, convert raw int16 to physical units using header.bitVolts.
        :param return_timestamps: when True, return sample timestamps in ms
        :param repress_output: when True, suppress print messages
        :return: data_matrix [n_channels, n_windows, nSamples]. If convert_microvolts=True, values are in **microvolts (µV)**.
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if len(channels) == 0 or channels is None:  # if no channels were provided
            channels = self.channelNumbers

        if not all([c in self.channelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            where_result = np.where((ts_array >= (start_time_ms[0][i] - self.recordLength)) &
                                    (ts_array < (start_time_ms[0][i] + window_ms)))
            p_single_trial_time_stamps = where_result[0] if len(where_result) > 0 else np.array([])
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                read_start_indices.append(p_single_trial_time_stamps)

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            if len(p_single_trial_time_stamps) > 0:
                single_trial_time_stamps = np.round(ts_array[p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
                # Get the number of records per trial & append to a list
                records_per_trial = len(single_trial_time_stamps)
            else:
                single_trial_time_stamps = np.array([])
                records_per_trial = 0
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                if repress_output is not True:
                    print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            c_file = self.oe_file_path / self.channel_files[channels[i] - 1]  # get path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * bytes_per_rec) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    # bytesPerRecCont is a scalar, blkBytesCont is an array
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    data_bytes = self.blkBytesCont[3] if len(self.blkBytesCont) > 3 else self.blkBytesCont[-1]
                    skip_size = int((bytes_per_rec - data_bytes) // 2)
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    try:
                        data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]),
                                                                    read_size + skip_size)
                    except ValueError:
                        print('There was a problem reshaping ...', data_plus_breaks)
                        if return_timestamps:
                            return None, None
                        else:
                            return None

                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                if repress_output is not True:
                    print(f'The requested data segment between {read_start_indices[j]} ms and '
                          f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                          f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_microvolts:
            # Open Ephys .continuous: header.bitVolts is microvolts per AD count for headstage channels.
            # raw_int16 * bitVolts = voltage in µV. Use per-channel scaling (channels can differ in gain).
            scale_uv_per_ch = np.array([
                self.MicrovoltsPerAD[np.where(self.channelNumbers == ch)[0][0]] for ch in channels
            ], dtype=float)
            data_matrix = data_matrix * scale_uv_per_ch[:, np.newaxis, np.newaxis]

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix

    def get_analog_data(self, channels, start_time_ms, window_ms, convert_microvolts=True, return_timestamps=True):
        """
        Retrieve continuous data from Open Ephys ADC/AUX .continuous files.

        Open Ephys convention: For ADC channels, header.bitVolts is in **volts per AD count**.
        This code stores bitVolts*1e6 as MicrovoltsPerADAnalog, so output is in µV when convert_microvolts=True.

        :param channels: channel numbers to sample from
        :param start_time_ms: window start times in ms
        :param window_ms: length of each window in ms
        :param convert_microvolts: when True, convert to physical units (output in **microvolts**, µV).
        :param return_timestamps: when True, return sample timestamps in ms
        :return: data_matrix [n_channels, n_windows, nSamples]; values in µV when convert_microvolts=True
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if not channels:  # if no channels were provided
            channels = self.analogChannelNumbers

        if not all([c in self.channelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            where_result = np.where((ts_array >= (start_time_ms[0][i] - self.recordLength)) &
                                    (ts_array < (start_time_ms[0][i] + window_ms)))
            p_single_trial_time_stamps = where_result[0] if len(where_result) > 0 else np.array([])
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                # Edge case: No matching timestamps found for this window
                # This can occur at recording boundaries or with sparse data
                # TODO: Improve handling of edge cases at recording boundaries
                read_start_indices.append(p_single_trial_time_stamps)

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            if len(p_single_trial_time_stamps) > 0:
                single_trial_time_stamps = np.round(ts_array[p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
                # Get the number of records per trial & append to a list
                records_per_trial = len(single_trial_time_stamps)
            else:
                single_trial_time_stamps = np.array([])
                records_per_trial = 0
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size, so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            c_file = self.oe_file_path / self.analog_files[channels[i] - 1]  # get path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * bytes_per_rec) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    # bytesPerRecCont is a scalar, blkBytesCont is an array
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    data_bytes = self.blkBytesCont[3] if len(self.blkBytesCont) > 3 else self.blkBytesCont[-1]
                    skip_size = int((bytes_per_rec - data_bytes) // 2)
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]), read_size + skip_size)
                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                print(f'The requested data segment between {read_start_indices[j]} ms and '
                      f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                      f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_microvolts:
            data_matrix = data_matrix * self.MicrovoltsPerADAnalog[-1]

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix

    def get_accel_data(self, channels, start_time_ms, window_ms, convert_microvolts=True, return_timestamps=True,
                       direct_paths_to_files=None):
        """
        Retrieve continuous data from Open Ephys AUX (accelerometer) .continuous files.

        ADC/AUX bitVolts are in volts per AD; stored as MicrovoltsPerADAnalog = bitVolts*1e6.
        Here we multiply by MicrovoltsPerADAnalog/1000 so output is in **millivolts (mV)** when convert_microvolts=True.

        :param channels: channel numbers to sample from
        :param start_time_ms: window start times in ms
        :param window_ms: length of each window in ms
        :param convert_microvolts: when True, output in **mV**
        :param return_timestamps: when True, return sample timestamps in ms
        :return: data_matrix [n_channels, n_windows, nSamples]; values in mV when convert_microvolts=True
        """
        window_samples = int(
            np.round(window_ms / self.sample_ms))  # round the time in ms to the nearest whole sample count
        n_windows = len(start_time_ms)  # get the number of start times provided
        start_time_ms = np.round(
            start_time_ms / self.sample_ms) * self.sample_ms  # round the start times to the nearest whole sample step
        window_ms = window_samples * self.sample_ms  # get the ms based length of the rounded window

        # deal with the channel numbers:
        if not channels:  # if no channels were provided
            channels = self.analogChannelNumbers

        if not all([c in self.analogChannelNumbers for c in channels]):  # if requested channels do not exist in the file
            raise ValueError('one or more of the entered channels does not exist in the recording!')
        n_ch = len(channels)

        # initialize some variables for the data extraction:
        # waveform matrix:
        data_matrix = np.zeros(shape=(int(window_samples), n_windows, n_ch),
                               dtype=self.blkCont['Types'][3],
                               order='F')
        # List to store the record indices for waveform extraction:
        p_rec_idx = []
        # List to store the indices where reading from the file should start (one per reading window):
        read_start_indices = []
        records_per_trial_list = []
        for i in range(n_windows):
            # find the relevant record blocks in the block list:
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            where_result = np.where((ts_array >= (start_time_ms[0][i] - self.recordLength)) &
                                    (ts_array < (start_time_ms[0][i] + window_ms)))
            p_single_trial_time_stamps = where_result[0] if len(where_result) > 0 else np.array([])
            try:
                # this collects the indices to start reading from
                read_start_indices.append(p_single_trial_time_stamps[0])
            except IndexError:
                # Edge case: No matching timestamps found for this window
                # This can occur at recording boundaries or with sparse data
                # TODO: Improve handling of edge cases at recording boundaries
                read_start_indices.append(p_single_trial_time_stamps)

            # Calculate time stamps in milliseconds based on sampling freq & record block length
            ts_array = self.allTimeStamps[0] if self.allTimeStamps.ndim > 1 else self.allTimeStamps
            if len(p_single_trial_time_stamps) > 0:
                single_trial_time_stamps = np.round(ts_array[p_single_trial_time_stamps] / self.sample_ms) * self.sample_ms
                # Get the number of records per trial & append to a list
                records_per_trial = len(single_trial_time_stamps)
            else:
                single_trial_time_stamps = np.array([])
                records_per_trial = 0
            records_per_trial_list.append(records_per_trial)

            # get the real time values for each sample index:
            time_idx = np.tile((np.arange(self.dataSamplesPerRecord) * self.sample_ms).reshape(-1, 1),
                               (1, records_per_trial)) + single_trial_time_stamps.reshape(1, -1)
            # Find time indices within the requested time window
            # (chunks are 1024 in size so they are usually cut for most time windows, result is as a boolean matrix)
            p_rec_idx.append((time_idx >= start_time_ms[0][i]) & (time_idx < (start_time_ms[0][i] + window_ms)))

            # Due to rounding issues, there may be an error when there is one sample too much -
            # in this case the last sample is removed
            if np.sum(p_rec_idx[i]) == window_samples + 1:
                print(f'sample removed for window #{i}')
                p_rec_idx[i][0, np.where(p_rec_idx[i][0, :] == 1)[0][0]] = False

        p_rec_idx = np.hstack(p_rec_idx)  # Concatenate record indices into a single array

        # now for the data extraction itself:
        for i in range(n_ch):  # iterate over channels
            data = np.zeros(p_rec_idx.shape, dtype=np.dtype('>i2'))  # Initialize the data array for a specific channel
            curr_rec = 0  # for this channel, initialize the record counter
            if direct_paths_to_files is None:
                c_file = self.oe_file_path / self.accel_files[channels[i] - 1]  # get path of current channel file
            else:
                c_file = self.oe_file_path / direct_paths_to_files[i] # get the direct path of current channel file
            with open(c_file, 'rb') as fid:  # open the file such that it will close when left alone
                for j in range(n_windows):  # Iterate over sampling windows
                    # use seek to go to the appropriate position in the file
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    fid.seek(int(self.headerSizeByte + (read_start_indices[j] * bytes_per_rec) + np.sum(
                        self.blkBytesCont[0:3])), 0)
                    # calculate the skip size, cut in half because each int16 is 2 bytes and the matlab
                    # function takes bytes as skip (which fromfile does not, uniform datatype)
                    # bytesPerRecCont is a scalar, blkBytesCont is an array
                    bytes_per_rec = self.bytesPerRecCont if isinstance(self.bytesPerRecCont, (int, float, np.integer)) else self.bytesPerRecCont[0]
                    data_bytes = self.blkBytesCont[3] if len(self.blkBytesCont) > 3 else self.blkBytesCont[-1]
                    skip_size = int((bytes_per_rec - data_bytes) // 2)
                    read_size = self.dataSamplesPerRecord
                    # calculate total element count to read:
                    total_bytes = (read_size + skip_size) * records_per_trial_list[j]
                    # read data from file in a single vector, including skip_data:
                    # (Notice datatype is non-flexible in this version of the function!!!)
                    data_plus_breaks = np.fromfile(fid, dtype=np.dtype('>i2'), count=total_bytes, sep='')
                    # reshape into an array with a column-per-record shape:
                    data_plus_breaks = data_plus_breaks.reshape(int(records_per_trial_list[j]), read_size + skip_size)
                    # slice the array to get rid of the skip_data at the end of each column (record):
                    clean_data = data_plus_breaks[:, : read_size]
                    # transpose and store the current_rec data:
                    data[:, curr_rec: curr_rec + records_per_trial_list[j]] = clean_data.T
                    curr_rec = curr_rec + records_per_trial_list[j]  # move forward to the next reading window
            # this loop exit closes the current channel file
            # vectorize the data from the channel and perform a boolean snipping of non-window samples:
            data_vec = data.T[p_rec_idx.T]

            # put the data in the final data_matrix waveform matrix:
            # check for end-of-recording exceedance :
            if len(data_vec) < int(window_samples) * n_windows:
                print(f'The requested data segment between {read_start_indices[j]} ms and '
                      f'{read_start_indices[j] + window_ms} ms exceeds the recording length, '
                      f'and will be 0-padded to fit the other windows')
                num_zeros = (int(window_samples) * n_windows) - len(data_vec)
                data_vec = np.pad(data_vec, (0, num_zeros), mode='constant')
            data_matrix[:, :, i] = data_vec.reshape(int(window_samples), n_windows, order='F')

        data_matrix = np.transpose(data_matrix, [2, 1, 0])

        if convert_microvolts:
            data_matrix = data_matrix * self.MicrovoltsPerADAnalog[-1] / 1000

        if return_timestamps:
            timestamps = np.tile(np.arange(window_samples) * self.sample_ms, (n_windows, 1))
            start_times = np.tile(start_time_ms.T, window_samples)
            timestamps = timestamps + start_times
            return data_matrix, timestamps
        else:
            return data_matrix
